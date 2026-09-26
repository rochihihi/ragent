import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from veripatch.config import Settings
from veripatch.studio_domain import StudioAction, StudioDecision
from veripatch.studio_model import (
    _OPENAI_PROTOCOL_CACHE,
    COMPAT_CHAT_PROMPT,
    STUDIO_PROMPT,
    StudioProviderModel,
    _native_decision,
    _native_tools,
    _plain_context,
    parse_studio_decision,
)


def test_native_tools_are_independent_strict_action_schemas() -> None:
    response_tools = _native_tools(responses_api=True)
    names = {item["name"] for item in response_tools}
    assert {
        "list_files",
        "search",
        "read",
        "edit",
        "apply_patch",
        "create",
        "run_tests",
        "run_command",
    } <= names
    assert "submit_decision" not in names
    assert all(item["strict"] is True for item in response_tools)
    assert all(item["parameters"]["additionalProperties"] is False for item in response_tools)
    mcp = next(item for item in response_tools if item["name"] == "mcp_call")
    assert mcp["parameters"]["properties"]["mcp_arguments"]["type"] == "string"
    assert _native_decision("mcp_call", {
        "rationale": "List project files", "mcp_tool": "list_project_files",
        "mcp_arguments": "{}",
    }).mcp_arguments == {}
    with pytest.raises(ValueError, match="MCP arguments must be a JSON object"):
        _native_decision("mcp_call", {
            "rationale": "Invalid arguments", "mcp_tool": "list_project_files",
            "mcp_arguments": "[]",
        })


def test_answer_phase_exposes_only_terminal_native_tools() -> None:
    tools = _native_tools(responses_api=True, allowed_actions={"respond", "finish"})
    assert {tool["name"] for tool in tools} == {"respond", "finish"}


def test_provider_uses_structured_model_intent_classifier() -> None:
    class ChatCompletions:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["response_format"] == {"type": "json_object"}
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"intent":"analysis","confidence":"high",'
                                '"requires_clarification":false,'
                                '"rationale":"用户只是在询问是否值得调整"}'
                            )
                        )
                    )
                ]
            )

    model = StudioProviderModel(
        "openai",
        Settings(model="gpt-test", openai_base_url="https://api.openai.com/v1"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions())),
    )
    assessment = asyncio.run(
        model.classify_intent([], "你觉得这个结构是不是该调整？")
    )

    assert assessment.intent == "analysis"
    assert assessment.confidence == "high"
    assert assessment.requires_clarification is False


def test_official_provider_ignores_saved_proxy_base_url(tmp_path) -> None:
    (tmp_path / "provider_connections.json").write_text(
        '{"openai": {"base_url": "https://proxy.example/v1"}}', encoding="utf-8"
    )
    model = StudioProviderModel(
        "openai_official",
        Settings(database_path=tmp_path / "state.sqlite3", openai_base_url="https://proxy.example/v1"),
        client=object(),
    )
    assert model.base_url == "https://api.openai.com/v1"


def test_official_openai_responses_uses_native_tools_and_combines_calls() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        async def create(self, **kwargs: object) -> SimpleNamespace:
            self.kwargs = kwargs
            return SimpleNamespace(
                id="resp-native-1",
                model="gpt-test",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="read",
                        arguments='{"rationale":"读取实现","path":"app.py"}',
                    ),
                    SimpleNamespace(
                        type="function_call",
                        name="search",
                        arguments='{"rationale":"查找引用","query":"calculate"}',
                    ),
                ],
            )

    _OPENAI_PROTOCOL_CACHE.clear()
    responses = FakeResponses()
    model = StudioProviderModel(
        "openai",
        Settings(model="gpt-test", openai_base_url="https://api.openai.com/v1"),
        client=SimpleNamespace(responses=responses),
    )
    model.base_url = "https://api.openai.com/v1"

    reply = asyncio.run(model.decide({"messages": []}))

    assert reply.decision.action is StudioAction.BATCH
    assert [item.action for item in reply.decision.actions] == [
        StudioAction.READ,
        StudioAction.SEARCH,
    ]
    assert responses.kwargs["tool_choice"] == "required"
    assert responses.kwargs["parallel_tool_calls"] is True
    assert len(responses.kwargs["tools"]) >= 10
    assert reply.protocol == "responses-tools"
    assert reply.response_id == "resp-native-1"


def test_official_answer_phase_does_not_offer_mcp_or_file_tools() -> None:
    class FakeResponses:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            assert {tool["name"] for tool in kwargs["tools"]} == {"respond", "finish"}
            return SimpleNamespace(
                id="answer-only", model="gpt-6-luna", usage=None,
                output=[SimpleNamespace(
                    type="function_call", name="respond",
                    arguments='{"rationale":"复用现有证据","message":"项目包含 app.py。","claims":[]}',
                )],
            )

    model = StudioProviderModel(
        "openai_official", Settings(model="gpt-6-luna"),
        client=SimpleNamespace(responses=FakeResponses()),
    )
    reply = asyncio.run(model.decide({"available_actions": ["respond", "finish"]}))
    assert reply.decision.action is StudioAction.RESPOND


def test_official_provider_returns_tool_result_with_original_call_id() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    id="resp-list", model="gpt-test", usage=None,
                    output=[SimpleNamespace(
                        type="function_call", call_id="call-list", name="list_files",
                        arguments='{"rationale":"查看文件"}',
                    )],
                )
            return SimpleNamespace(
                id="resp-answer", model="gpt-test", usage=None,
                output=[SimpleNamespace(
                    type="function_call", call_id="call-answer", name="respond",
                    arguments='{"rationale":"结果足够","message":"有 app.py。","claims":[]}',
                )],
            )

    responses = FakeResponses()
    model = StudioProviderModel(
        "openai_official", Settings(model="gpt-test"),
        client=SimpleNamespace(responses=responses),
    )
    first = asyncio.run(model.decide({
        "session_id": "s1", "turn_observation_start": 0,
        "latest_tool_result": None,
    }))
    second = asyncio.run(model.decide({
        "session_id": "s1", "turn_observation_start": 0,
        "latest_tool_result": {
            "observation_id": 0, "kind": "files", "payload": {"files": ["app.py"]},
        },
    }))
    assert first.decision.action is StudioAction.LIST_FILES
    assert second.decision.action is StudioAction.RESPOND
    assert responses.requests[1]["previous_response_id"] == "resp-list"
    assert responses.requests[1]["input"][0]["call_id"] == "call-list"
    assert "app.py" in responses.requests[1]["input"][0]["output"]
    assert {tool["name"] for tool in responses.requests[1]["tools"]} > {"list_files", "respond"}
    asyncio.run(model.decide({
        "session_id": "s1", "turn_observation_start": 1,
        "latest_tool_result": {"observation_id": 0, "kind": "files"},
    }))
    assert "previous_response_id" not in responses.requests[2]


def test_official_provider_restores_tool_call_after_permission_pause() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    id="resp-launch", model="gpt-test", usage=None,
                    output=[SimpleNamespace(
                        type="function_call", call_id="call-launch", name="run_command",
                        arguments='{"rationale":"打开文件","command":["start","index.html"]}',
                    )],
                )
            return SimpleNamespace(
                id="resp-final", model="gpt-test", usage=None,
                output=[SimpleNamespace(
                    type="function_call", call_id="call-final", name="respond",
                    arguments='{"rationale":"启动请求已发出","message":"已发出打开请求，尚未确认窗口。","claims":[]}',
                )],
            )

    responses = FakeResponses()
    settings = Settings(model="gpt-test")
    client = SimpleNamespace(responses=responses)
    before_pause = StudioProviderModel("openai_official", settings, client=client)
    first = asyncio.run(before_pause.decide({
        "session_id": "s1", "turn_observation_start": 0, "latest_tool_result": None,
    }))
    assert first.tool_continuation is not None

    after_approval = StudioProviderModel("openai_official", settings, client=client)
    after_approval.restore_tool_continuation(
        first.tool_continuation, session_id="s1", turn_observation_start=0,
    )
    second = asyncio.run(after_approval.decide({
        "session_id": "s1", "turn_observation_start": 0,
        "latest_tool_result": {
            "observation_id": 0, "kind": "command",
            "payload": {"launch_state": "dispatched", "window_confirmed": None},
        },
    }))
    assert second.decision.action is StudioAction.RESPOND
    assert responses.requests[1]["previous_response_id"] == "resp-launch"
    assert responses.requests[1]["input"][0]["call_id"] == "call-launch"
    assert "dispatched" in responses.requests[1]["input"][0]["output"]


def test_compatible_chat_returns_tool_result_with_original_call_id() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                call = SimpleNamespace(
                    id="call-list", function=SimpleNamespace(
                        name="list_files", arguments='{"rationale":"查看文件"}',
                    ),
                )
            else:
                call = SimpleNamespace(
                    id="call-answer", function=SimpleNamespace(
                        name="respond",
                        arguments='{"rationale":"结果足够","message":"有 app.py。","claims":[]}',
                    ),
                )
            return SimpleNamespace(
                model="proxy-test", usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
            )

    _OPENAI_PROTOCOL_CACHE.clear()
    completions = FakeCompletions()
    model = StudioProviderModel(
        "openai", Settings(model="proxy-test", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    asyncio.run(model.decide({
        "session_id": "s1", "turn_observation_start": 0, "latest_tool_result": None,
    }))
    second = asyncio.run(model.decide({
        "session_id": "s1", "turn_observation_start": 0,
        "latest_tool_result": {
            "observation_id": 0, "kind": "files", "payload": {"files": ["app.py"]},
        },
    }))
    assert second.decision.action is StudioAction.RESPOND
    assert completions.requests[1]["messages"][1]["tool_calls"][0]["id"] == "call-list"
    assert completions.requests[1]["messages"][2]["tool_call_id"] == "call-list"
    assert "app.py" in completions.requests[1]["messages"][2]["content"]
    asyncio.run(model.decide({
        "session_id": "s1", "turn_observation_start": 1,
        "latest_tool_result": {"observation_id": 0, "kind": "files"},
    }))
    assert len(completions.requests[2]["messages"]) == 1


def test_studio_prompt_uses_ragent_identity() -> None:
    assert "You are RAgent" in STUDIO_PROMPT
    assert "never merely paste code" in STUDIO_PROMPT


def test_code_handoff_is_rejected_for_agent_execution() -> None:
    decision = StudioDecision(
        action=StudioAction.RESPOND,
        rationale="给出实现",
        message="请将以下代码保存为 calculator.py\n```python\nprint(1)\n```",
    )
    assert StudioProviderModel._is_code_handoff(decision) is True
    explanation = StudioDecision(
        action=StudioAction.RESPOND,
        rationale="解释概念",
        message="这个项目使用事件驱动架构。",
    )
    assert StudioProviderModel._is_code_handoff(explanation) is False
    assert "Never call yourself VeriPatch" in STUDIO_PROMPT


def test_plain_language_python_command_is_not_guessed_as_tool_action() -> None:
    with pytest.raises(ValidationError):
        parse_studio_decision("文件已修改。请运行 python missing.py 验证。")


def test_false_tool_unavailable_failure_is_rejected() -> None:
    decision = StudioDecision(
        action=StudioAction.FAIL,
        rationale="当前会话未提供文件创建、编辑或命令执行工具。",
        message="无法完成任务。",
    )
    assert StudioProviderModel._is_false_tool_unavailable_failure(decision) is True
    real_failure = StudioDecision(
        action=StudioAction.FAIL,
        rationale="项目要求的外部服务当前不可用。",
        message="需要稍后重试。",
    )
    assert StudioProviderModel._is_false_tool_unavailable_failure(real_failure) is False
    ordinary_reply = StudioDecision(
        action=StudioAction.RESPOND,
        rationale="无法执行",
        message="当前环境没有可用的文件编辑和命令执行工具。",
    )
    assert StudioProviderModel._is_false_tool_unavailable_failure(ordinary_reply) is True


def test_unverified_changed_files_cannot_be_handed_back_to_user() -> None:
    decision = StudioDecision(
        action=StudioAction.RESPOND,
        rationale="让用户自行验证",
        message="请运行 python probe.py。",
    )
    context = {
        "changed_files": ["probe.py"],
        "verification": {"verification_passed": False},
    }

    assert StudioProviderModel._is_unverified_execution_handoff(decision, context) is True
    context["verification"] = {"verification_passed": True}
    assert StudioProviderModel._is_unverified_execution_handoff(decision, context) is False


def test_python_execution_handoff_is_promoted_to_tool_action() -> None:
    decision = StudioDecision(
        action=StudioAction.RESPOND,
        rationale="让用户验证",
        message="请在项目目录运行 python probe.py；预期输出正常。",
    )
    context = {
        "changed_files": ["probe.py"],
        "verification": {"verification_passed": False},
    }

    promoted = StudioProviderModel._promote_execution_handoff(decision, context)

    assert promoted.action is StudioAction.RUN_COMMAND
    assert promoted.command == ["python", "probe.py"]


def test_studio_prompt_does_not_add_product_content_moderation() -> None:
    assert "fictional software fixtures as synthetic test data" in STUDIO_PROMPT
    assert "within the model provider's capabilities" in STUDIO_PROMPT
    assert "limits\npersistent storage only" in STUDIO_PROMPT


def test_parse_studio_decision_accepts_pure_json() -> None:
    decision = parse_studio_decision('{"action":"list_files","rationale":"查看结构"}')

    assert decision.action is StudioAction.LIST_FILES


def test_proxy_context_is_rendered_without_json_envelope() -> None:
    rendered = _plain_context(
        {"messages": [{"role": "user", "content": "inspect"}], "changed_files": []}
    )

    assert "messages:" in rendered
    assert "role: user" in rendered
    assert "content: inspect" in rendered
    assert "{" not in rendered


def test_parse_studio_decision_extracts_json_after_explanation() -> None:
    decision = parse_studio_decision(
        "我将先查看项目结构。\n```json\n"
        '{"action":"list_files","rationale":"先确认项目文件","message":null}\n```'
    )

    assert decision.action is StudioAction.LIST_FILES
    assert decision.rationale == "先确认项目文件"


def test_parse_studio_decision_still_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        parse_studio_decision("我将查看项目，但这里没有结构化动作。")


def test_protocol_parse_error_preserves_safe_raw_response_evidence() -> None:
    response = SimpleNamespace(id="req-proxy-42")

    with pytest.raises(ValueError) as caught:
        StudioProviderModel._parse_response_decision(
            "我会先看看代码。credential=sk-secret123456", response
        )

    error: Any = caught.value
    assert error.response_id == "req-proxy-42"
    assert error.response_shape == "普通文本"
    preview = error.raw_response_preview
    assert "我会先看看代码" in preview
    assert "sk-secret123456" not in preview
    assert "[密钥已隐藏]" in preview


def test_studio_provider_retries_plain_text_as_structured_json() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(self, **kwargs: object) -> SimpleNamespace:
            self.calls.append(kwargs)
            output = (
                "我先了解项目结构。"
                if len(self.calls) == 1
                else '{"action":"list_files","rationale":"查看项目结构"}'
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=output))],
                model="deepseek-v4-flash",
            )

    responses = FakeResponses()
    model = StudioProviderModel(
        "deepseek",
        Settings(deepseek_model="deepseek-v4-flash"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=responses)),
    )

    reply = asyncio.run(model.decide({"messages": []}))

    assert reply.decision.action is StudioAction.LIST_FILES
    assert len(responses.calls) == 2
    assert responses.calls[1]["response_format"] == {"type": "json_object"}


def test_openai_proxy_uses_chat_directly_and_caches_protocol() -> None:
    class ResponsesMustNotBeCalled:
        calls = 0

        async def parse(self, **_kwargs: object) -> None:
            self.calls += 1
            raise AssertionError("third-party gateways must not be probed with Responses")

    class ChatCompletions:
        calls = 0
        kwargs: list[dict[str, object]] = []

        async def create(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            self.kwargs.append(kwargs)
            message = SimpleNamespace(
                content='{"action":"respond","rationale":"回答身份问题","message":"当前模型"}'
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], model="proxy-model", usage=None
            )

    _OPENAI_PROTOCOL_CACHE.clear()
    responses = ResponsesMustNotBeCalled()
    completions = ChatCompletions()
    client = SimpleNamespace(
        responses=responses,
        chat=SimpleNamespace(completions=completions),
    )
    settings = Settings(
        model="proxy-model",
        openai_base_url="https://proxy.example/v1",
    )

    first = asyncio.run(StudioProviderModel("openai", settings, client=client).decide({}))
    second = asyncio.run(StudioProviderModel("openai", settings, client=client).decide({}))

    assert first.decision.action is StudioAction.RESPOND
    assert second.decision.message == "当前模型"
    assert responses.calls == 0
    assert completions.calls == 2
    assert _OPENAI_PROTOCOL_CACHE["https://proxy.example/v1"] == "chat-json"
    assert completions.kwargs[0]["reasoning_effort"] == "medium"
    tool_names = {item["function"]["name"] for item in completions.kwargs[0]["tools"]}
    assert {"read", "edit", "run_command", "finish"} <= tool_names
    assert completions.kwargs[0]["tool_choice"] == "required"
    assert completions.kwargs[1]["response_format"] == {"type": "json_object"}
    prompt = completions.kwargs[0]["messages"][0]["content"]
    assert "content must contain the complete new file" in prompt
    messages = completions.kwargs[0]["messages"]
    assert isinstance(messages, list)
    assert COMPAT_CHAT_PROMPT in messages[0]["content"]


def test_openai_proxy_accepts_native_tool_decision() -> None:
    class ChatCompletions:
        async def create(self, **_kwargs: object) -> SimpleNamespace:
            function = SimpleNamespace(
                name="submit_decision",
                arguments='{"action":"read","rationale":"读取实现","path":"app.py"}',
            )
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(function=function)])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], model="proxy-model", usage=None
            )

    _OPENAI_PROTOCOL_CACHE.clear()
    client = SimpleNamespace(
        responses=SimpleNamespace(),
        chat=SimpleNamespace(completions=ChatCompletions()),
    )
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=client,
    )

    reply = asyncio.run(model.decide({}))

    assert reply.decision.action is StudioAction.READ
    assert reply.decision.path == "app.py"
    assert _OPENAI_PROTOCOL_CACHE["https://proxy.example/v1"] == "chat-tools"


def test_openai_proxy_combines_multiple_independent_native_tool_calls() -> None:
    class ChatCompletions:
        async def create(self, **_kwargs: object) -> SimpleNamespace:
            calls = [
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="read",
                        arguments='{"rationale":"读取实现","path":"app.py"}',
                    )
                ),
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="search",
                        arguments='{"rationale":"查找调用","query":"main"}',
                    )
                ),
            ]
            message = SimpleNamespace(content=None, tool_calls=calls)
            return SimpleNamespace(
                id="chat-native-2",
                choices=[SimpleNamespace(message=message)],
                model="proxy-model",
                usage=None,
            )

    _OPENAI_PROTOCOL_CACHE.clear()
    client = SimpleNamespace(
        responses=SimpleNamespace(),
        chat=SimpleNamespace(completions=ChatCompletions()),
    )
    reply = asyncio.run(
        StudioProviderModel(
            "openai",
            Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
            client=client,
        ).decide({})
    )

    assert reply.decision.action is StudioAction.BATCH
    assert [item.action for item in reply.decision.actions] == [
        StudioAction.READ,
        StudioAction.SEARCH,
    ]
    assert reply.protocol == "chat-tools"


def test_openai_proxy_empty_tool_response_downgrades_to_json() -> None:
    class ChatCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(self, **kwargs: object) -> SimpleNamespace:
            self.calls.append(kwargs)
            content = (
                ""
                if len(self.calls) == 1
                else '{"action":"create","rationale":"创建文件","path":"app.py",'
                '"content":"print(1)\\n"}'
            )
            message = SimpleNamespace(content=content, tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], model="proxy-model", usage=None
            )

    _OPENAI_PROTOCOL_CACHE.clear()
    completions = ChatCompletions()
    client = SimpleNamespace(
        responses=SimpleNamespace(),
        chat=SimpleNamespace(completions=completions),
    )
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=client,
    )

    reply = asyncio.run(model.decide({}))

    assert reply.decision.action is StudioAction.CREATE
    assert len(completions.calls) == 2
    assert "tools" in completions.calls[0]
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    assert _OPENAI_PROTOCOL_CACHE["https://proxy.example/v1"] == "chat-json"


def test_raw_proxy_request_identifies_ragent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def read(self) -> bytes:
            return b""

        def json(self) -> dict[str, object]:
            return {
                "model": "proxy-model",
                "choices": [{"message": {"content": '{"action":"respond"}'}}],
            }

    def stream(_method: str, _url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("veripatch.studio_model.httpx.stream", stream)
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )
    model.api_key = "secret"

    model._raw_compatible_chat({"model": "proxy-model", "messages": []})

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["User-Agent"] == "RAgent/3.0.0"
    assert headers["X-Title"] == "RAgent"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["stream"] is True


def test_raw_proxy_leaves_transient_protocol_retry_to_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def read(self) -> bytes:
            return b""

        def json(self) -> dict[str, object]:
            return {
                "model": "proxy-model",
                "choices": [{"message": {"content": '{"action":"respond"}'}}],
            }

    def stream(_method: str, _url: str, **_kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("server disconnected")

    monkeypatch.setattr("veripatch.studio_model.httpx.stream", stream)
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )
    model.api_key = "secret"

    with pytest.raises(httpx.RemoteProtocolError):
        model._raw_compatible_chat({"model": "proxy-model", "messages": []})

    assert calls == 1


def test_raw_proxy_leaves_transient_http_retry_to_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            nonlocal calls
            request = httpx.Request("POST", "https://proxy.example/v1/chat/completions")
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError("bad gateway", request=request, response=response)

        def read(self) -> bytes:
            return b""

        def json(self) -> dict[str, object]:
            return {
                "model": "proxy-model",
                "choices": [{"message": {"content": '{"action":"respond"}'}}],
            }

    def stream(_method: str, _url: str, **_kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr("veripatch.studio_model.httpx.stream", stream)
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )
    model.api_key = "secret"

    with pytest.raises(httpx.HTTPStatusError):
        model._raw_compatible_chat({"model": "proxy-model", "messages": []})

    assert calls == 1


def test_raw_proxy_request_collects_streamed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        headers = {"content-type": "text/event-stream; charset=utf-8"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return [
                "data: "
                + json.dumps(
                    {
                        "model": "proxy-model",
                        "choices": [{"delta": {"content": '{"action":"'}}],
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "choices": [{"delta": {"content": 'respond"}'}}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                    }
                ),
                "data: [DONE]",
            ]

    monkeypatch.setattr(
        "veripatch.studio_model.httpx.stream",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )
    model.api_key = "secret"

    response = model._raw_compatible_chat({"model": "proxy-model", "messages": []})

    assert response.choices[0].message.content == '{"action":"respond"}'
    assert response.usage.prompt_tokens == 4


def test_official_deepseek_waits_for_final_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        headers = {"content-type": "text/event-stream"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> Any:
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": '{"action":"respond","message":"好"}'}}]}
            )
            yield "data: " + json.dumps(
                {"choices": [], "usage": {"prompt_tokens": 321, "completion_tokens": 8}}
            )
            yield "data: [DONE]"

    def stream(_method: str, _url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("veripatch.studio_model.httpx.stream", stream)
    model = StudioProviderModel("deepseek", Settings(), client=SimpleNamespace())
    model.api_key = "secret"

    response = model._raw_compatible_chat({"model": model.model_name, "messages": []})

    assert response.usage.prompt_tokens == 321
    assert captured["json"]["stream_options"] == {"include_usage": True}


def test_raw_proxy_request_collects_streamed_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        headers = {"content-type": "text/event-stream"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> Any:
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "submit_decision",
                                            "arguments": '{"action":"read",',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": ('"rationale":"inspect","path":"app.py"}')
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr(
        "veripatch.studio_model.httpx.stream",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )
    model.api_key = "secret"

    response = model._raw_compatible_chat({"model": "proxy-model", "messages": []})
    decision = model._chat_message_decision(response.choices[0].message, response)

    assert decision.action is StudioAction.READ
    assert decision.path == "app.py"


def test_raw_proxy_stops_after_a_complete_decision_without_waiting_for_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        headers = {"content-type": "text/event-stream"}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> Any:
            yield "data: " + json.dumps(
                {
                    "model": "proxy-model",
                    "choices": [
                        {
                            "delta": {
                                "content": json.dumps(
                                    {
                                        "action": "respond",
                                        "rationale": "直接回答",
                                        "message": "完成",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                },
                ensure_ascii=False,
            )
            raise TimeoutError("gateway never closed its stream")

    monkeypatch.setattr(
        "veripatch.studio_model.httpx.stream",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )
    model.api_key = "secret"

    response = model._raw_compatible_chat({"model": "proxy-model", "messages": []})

    assert json.loads(response.choices[0].message.content)["message"] == "完成"


def test_compatible_response_accepts_responses_style_output() -> None:
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )

    response = model._compatible_response(
        {
            "model": "proxy-model",
            "choices": [],
            "output": [{"content": [{"type": "output_text", "text": '{"action":"respond"}'}]}],
        }
    )

    assert response.choices[0].message.content == '{"action":"respond"}'


def test_compatible_response_rejects_empty_choices_without_index_error() -> None:
    model = StudioProviderModel(
        "openai",
        Settings(model="proxy-model", openai_base_url="https://proxy.example/v1"),
        client=SimpleNamespace(),
    )

    with pytest.raises(httpx.RemoteProtocolError, match="returned no choices"):
        model._compatible_response({"choices": []})
