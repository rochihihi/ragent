import asyncio
from types import SimpleNamespace

import pytest

from veripatch.config import Settings
from veripatch.domain import ActionKind, IssueSpec
from veripatch.models.base import ModelContext
from veripatch.models.deepseek import DeepSeekModel


def _context() -> ModelContext:
    return ModelContext(
        issue=IssueSpec(issue_id="1", title="Bug", description="Repair it"),
        step=1,
        changed_files=[],
        index_summary={"symbols": 1},
        recent_observations=[],
    )


def _decision_json() -> str:
    return '{"action":"search","rationale":"Find code","query":"target"}'


class FakeResponses:
    def __init__(self, contents: list[str], *, error: Exception | None = None) -> None:
        self.contents = contents
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = self.contents.pop(0)
        return SimpleNamespace(
            id="resp_123",
            model="deepseek-v4-flash-202608",
            output_text=content,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=25,
                input_tokens_details=SimpleNamespace(cached_tokens=40),
                output_tokens_details=SimpleNamespace(reasoning_tokens=8),
            ),
        )


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="chat_123",
            model="deepseek-v4-pro",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=77,
                completion_tokens=19,
                prompt_cache_hit_tokens=11,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=6),
            ),
        )


def test_flash_uses_responses_json_schema_and_maps_usage() -> None:
    responses = FakeResponses([_decision_json()])
    client = SimpleNamespace(responses=responses)
    reply = asyncio.run(DeepSeekModel(Settings(), client=client).decide(_context()))
    assert reply.decision.action is ActionKind.SEARCH
    assert (reply.input_tokens, reply.cached_input_tokens) == (100, 40)
    assert (reply.output_tokens, reply.reasoning_tokens) == (25, 8)
    assert reply.model == "deepseek-v4-flash-202608"
    assert reply.request_id == "resp_123"
    request = responses.calls[0]
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["reasoning"] == {"effort": "medium"}
    assert "Simplified Chinese" in request["instructions"]


def test_flash_retries_empty_or_invalid_output_once() -> None:
    responses = FakeResponses(["", _decision_json()])
    reply = asyncio.run(
        DeepSeekModel(Settings(), client=SimpleNamespace(responses=responses)).decide(_context())
    )
    assert reply.decision.query == "target"
    assert reply.input_tokens == 200
    assert len(responses.calls) == 2
    assert "previous response was invalid" in responses.calls[1]["input"]

    invalid = FakeResponses(["not json", "{}"])
    with pytest.raises(RuntimeError, match="invalid AgentDecision"):
        asyncio.run(
            DeepSeekModel(Settings(), client=SimpleNamespace(responses=invalid)).decide(_context())
        )


def test_flash_normalizes_null_edits_and_recovers_trailing_json() -> None:
    null_edits = '{"action":"search","rationale":"Find code","query":"target","edits":null}'
    reply = asyncio.run(
        DeepSeekModel(
            Settings(), client=SimpleNamespace(responses=FakeResponses([null_edits]))
        ).decide(_context())
    )
    assert reply.decision.edits == []

    trailing = (
        '{"action":"search","rationale":"Missing query"}\n'
        '{"action":"read","rationale":"Inspect code","path":"pagination.py"}'
    )
    reply = asyncio.run(
        DeepSeekModel(
            Settings(), client=SimpleNamespace(responses=FakeResponses([trailing]))
        ).decide(_context())
    )
    assert reply.decision.action is ActionKind.READ
    assert reply.decision.path == "pagination.py"


def test_flash_wraps_provider_error() -> None:
    responses = FakeResponses([], error=TimeoutError("timed out"))
    with pytest.raises(RuntimeError, match="DeepSeek API failed"):
        asyncio.run(
            DeepSeekModel(Settings(), client=SimpleNamespace(responses=responses)).decide(
                _context()
            )
        )


def test_pro_uses_chat_json_output_and_normalizes_reasoning_effort() -> None:
    completions = FakeCompletions(_decision_json())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = Settings(deepseek_model="deepseek-v4-pro", reasoning_effort="medium")
    reply = asyncio.run(DeepSeekModel(settings, client=client).decide(_context()))
    assert reply.input_tokens == 77
    assert reply.cached_input_tokens == 11
    assert reply.reasoning_tokens == 6
    assert completions.kwargs is not None
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["reasoning_effort"] == "high"


def test_response_text_reads_nested_output() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=_decision_json())],
            )
        ]
    )
    assert DeepSeekModel._response_text(response) == _decision_json()
