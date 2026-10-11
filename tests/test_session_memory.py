import asyncio
import json

import pytest

from veripatch.studio_agent import StudioAgent
from veripatch.studio_domain import (
    StudioAction,
    StudioDecision,
    StudioMemoryUpdate,
    StudioMessage,
    StudioObservation,
    StudioReply,
    StudioSession,
)
from veripatch.studio_memory import project_memory, redact_memory_text, remember
from veripatch.studio_model import _native_decision, _native_tools
from veripatch.studio_store import StudioStore


def make_session(tmp_path, session_id="memory"):
    return StudioSession(
        session_id=session_id, repo_root=str(tmp_path), provider="openai", model="test",
        reasoning_effort="low",
    )


def test_relevant_old_notes_survive_recent_noise_without_mutating_storage(tmp_path):
    session = make_session(tmp_path)
    remember(session, "facts", "billing.py 使用 Decimal 计算退款", source="model_note")
    remember(session, "hypotheses", "退款舍入可能出错", source="model_note")
    for index in range(65):
        remember(session, "facts", f"已阅读其他文件 item_{index}.py", source="model_note")
    before = session.model_dump_json()
    memory = project_memory(session, "检查 billing.py 的退款计算")
    assert memory["facts"][0] == "billing.py 使用 Decimal 计算退款"
    assert memory["hypotheses"] == ["退款舍入可能出错"]
    assert memory["provenance"]["facts"][0]["source"] == "model_note"
    assert session.model_dump_json() == before


def test_memory_is_restored_only_for_the_same_session(tmp_path):
    session = make_session(tmp_path, "old")
    remember(session, "facts", "项目采用 Vue", source="model_note")
    store = StudioStore(tmp_path / "state.db")
    store.save(session, "test", {})
    restored = store.load("old")
    assert project_memory(restored)["facts"] == ["项目采用 Vue"]
    assert restored.memory.entries[0].source == "model_note"
    new = make_session(tmp_path, "new")
    assert project_memory(new)["facts"] == []
    assert project_memory(new)["scope"] == "current_session_only"


def test_model_can_replace_its_own_note_but_not_tool_evidence(tmp_path):
    session = make_session(tmp_path)
    session.messages.append(StudioMessage(role="user", content="改为使用 Vue"))
    remember(session, "facts", "界面使用 React", source="model_note")
    observation = StudioObservation(kind="read", summary="已读取 package.json", payload={})
    session.observations.append(observation)
    remember(
        session, "facts", "曾读取 package.json", source="tool_observation", observation_index=0,
    )
    update = StudioMemoryUpdate(
        forget_facts=["界面使用 React", "曾读取 package.json"], facts=["界面使用 Vue"],
    )
    StudioAgent._apply_memory_update(
        session, StudioDecision(action="respond", rationale="更新笔记", memory_update=update),
    )
    assert session.memory.facts == ["曾读取 package.json", "界面使用 Vue"]
    assert session.observations == [observation]
    assert session.messages[0].content == "改为使用 Vue"
    sources = project_memory(session)["provenance"]["facts"]
    assert any(source.get("observation_index") == 0 for source in sources)


def test_duplicate_model_note_cannot_overwrite_tool_provenance(tmp_path):
    session = make_session(tmp_path)
    remember(session, "facts", "验证通过", source="tool_observation", observation_index=2)
    remember(session, "facts", "验证通过", source="model_note")
    assert len(session.memory.entries) == 1
    assert session.memory.entries[0].source == "tool_observation"
    assert session.memory.entries[0].observation_index == 2


@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmnop", "OPENAI_API_KEY=abcdefghijk", '"api_key": "abcdefghi"',
    "Authorization: Bearer abcdefghi", "password=abcdefghi", "refresh_token=abcdefghi",
])
def test_credentials_redacted_before_notes_are_persisted(tmp_path, secret):
    session = make_session(tmp_path)
    remember(session, "facts", secret, source="model_note")
    assert "[REDACTED]" in session.model_dump_json()
    assert "abcdefghi" not in session.model_dump_json()
    assert redact_memory_text(secret) == session.memory.facts[0]


def test_legacy_memory_is_readable_but_never_promoted_to_verified(tmp_path):
    session = make_session(tmp_path)
    session.memory.facts = ["旧版记录：验证通过 api_key=abcdefghijk"]
    memory = project_memory(session)
    assert "abcdefghijk" not in json.dumps(memory)
    assert memory["provenance"]["facts"] == [{"source": "legacy"}]
    assert "unverified" in memory["status"]


def test_projection_and_provenance_have_joint_bounds(tmp_path):
    session = make_session(tmp_path)
    for category in ("facts", "hypotheses", "relevant_files", "failures"):
        for index in range(100):
            remember(session, category, f"{index}: " + "中" * 800, source="model_note")
    assert len(session.memory.entries) == 240
    for budget in (1_000, 3_000, 9_000):
        memory = project_memory(session, max_chars=budget)
        assert len(json.dumps(memory, ensure_ascii=False)) <= budget
        for category in ("facts", "hypotheses", "relevant_files", "failures"):
            assert len(memory[category]) == len(memory["provenance"][category])
    restored = StudioSession.model_validate_json(session.model_dump_json())
    assert restored.memory == session.memory


@pytest.mark.parametrize("responses_api", [True, False])
def test_native_tools_expose_strict_nullable_memory_updates(responses_api):
    for tool in _native_tools(responses_api=responses_api):
        function = tool if responses_api else tool["function"]
        parameters = function["parameters"]
        assert "memory_update" in parameters["required"]
        schema = parameters["properties"]["memory_update"]
        assert {"type": "null"} in schema["anyOf"]
        update = next(item for item in schema["anyOf"] if item["type"] == "object")
        assert update["additionalProperties"] is False
        assert set(update["required"]) == set(update["properties"])
        assert "forget_hypotheses" in update["properties"]


def test_native_decision_parses_memory_update_without_changing_action():
    decision = _native_decision("respond", json.dumps({
        "rationale": "回答", "message": "收到", "claims": [],
        "memory_update": {
            "facts": ["使用 Vue"], "hypotheses": [], "relevant_files": [],
            "forget_facts": ["使用 React"], "forget_hypotheses": [],
        },
    }))
    assert decision.action is StudioAction.RESPOND
    assert decision.memory_update.facts == ["使用 Vue"]
    assert decision.memory_update.forget_facts == ["使用 React"]
    no_update = _native_decision("read", json.dumps({
        "rationale": "读取", "path": "README.md", "memory_update": None,
    }))
    assert no_update.memory_update is None


def test_generated_handoff_redacts_credentials_without_editing_original_history(tmp_path):
    class Model:
        async def summarize_context(self, source):
            return {"summary": "项目使用 Vue，api_key=abcdefghijk"}

    session = make_session(tmp_path)
    session.messages = [
        StudioMessage(role="user", content="历史讨论 " * 500) for _ in range(12)
    ]
    context = {
        "messages": [item.model_dump(mode="json") for item in session.messages],
        "current_request": "继续解释，不要修改文件",
        "structured_memory": project_memory(session),
    }
    original = session.model_dump_json()
    store = StudioStore(tmp_path / "state.db")
    agent = StudioAgent(Model(), store, max_context_tokens=10_000)
    result = asyncio.run(agent._summarize_model_context(session, context, "继续解释，不要修改文件"))
    assert "abcdefghijk" not in session.model_context_summary
    assert "[REDACTED]" in result["handoff_summary"]["text"]
    assert session.messages == StudioSession.model_validate_json(original).messages
    assert store.load(session.session_id).summary_version == 1


@pytest.mark.parametrize("payload,kind", [
    ({"exit_code": None}, "command"),
    ({"exit_code": 0, "timed_out": True}, "command"),
    ({}, "tool_error"),
    ({"exit_code": 0}, "tool_error"),
])
def test_pending_timeout_or_tool_errors_never_become_success_notes(tmp_path, payload, kind):
    session = make_session(tmp_path)
    observation = StudioObservation(kind=kind, summary="运行状态", payload=payload)
    session.observations.append(observation)
    StudioAgent._learn_from_observation(session, StudioAction.RUN_COMMAND, observation)
    assert not any("验证 通过" in fact for fact in session.memory.facts)
    if kind == "command" and payload.get("exit_code") is None:
        assert not session.memory.failures
    else:
        assert session.memory.failures
    assert all(entry.observation_index == 0 for entry in session.memory.entries)


def test_failed_read_is_not_remembered_as_a_success(tmp_path):
    session = make_session(tmp_path)
    observation = StudioObservation(
        kind="tool_error", summary="找不到文件", payload={"path": "missing.py"},
    )
    session.observations.append(observation)
    StudioAgent._learn_from_observation(session, StudioAction.READ, observation)
    assert not session.memory.facts
    assert session.memory.failures == ["找不到文件"]


def test_agent_receives_relevant_session_notes_on_the_real_context_path(tmp_path):
    class Model:
        contexts = []

        async def decide(self, context):
            self.contexts.append(context)
            return StudioReply(decision=StudioDecision(
                action="respond", rationale="回答", message="已检查记忆",
                memory_update=StudioMemoryUpdate(facts=["本轮已讨论退款计算"]),
            ))

    session = make_session(tmp_path)
    remember(session, "facts", "退款计算使用 Decimal", source="model_note")
    for index in range(45):
        remember(session, "facts", f"无关记录 {index}", source="model_note")
    model = Model()
    store = StudioStore(tmp_path / "state.db")
    asyncio.run(StudioAgent(model, store).handle(session, "退款计算使用什么类型？"))
    assert model.contexts[0]["structured_memory"]["facts"][0] == "退款计算使用 Decimal"
    assert "本轮已讨论退款计算" in store.load(session.session_id).memory.facts
