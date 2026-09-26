import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from veripatch.api import _project_root, create_app
from veripatch.config import Settings
from veripatch.domain import FileEdit, TestOutcome as RunnerOutcome
from veripatch.studio_agent import StudioAgent, _requests_code_change, context_limit_for_model
from veripatch.studio_api import (
    _is_pure_bulk_delete_request,
    _latest_task_message,
    _restore_before_from_unified_diff,
)
from veripatch.studio_domain import (
    PlanStatus,
    ResponseStyle,
    SemanticIntentAssessment,
    StudioAction,
    StudioDecision,
    StudioMemoryUpdate,
    StudioMessage,
    StudioObservation,
    StudioPermissionRequest,
    StudioPlanItem,
    StudioReply,
    StudioRequirement,
    StudioSession,
    StudioTaskContract,
    StudioTaskState,
    VerificationMode,
)
from veripatch.studio_intent import classify_intent
from veripatch.studio_store import StudioStore
from veripatch.studio_tools import is_detached_launch
from veripatch.workspace import SafeWorkspace


class SequenceStudioModel:
    def __init__(self, decisions: list[StudioDecision]) -> None:
        self.decisions = decisions

    async def decide(self, context: dict[str, object]) -> StudioReply:
        return StudioReply(decision=self.decisions.pop(0), input_tokens=10, output_tokens=2)


class CapturingStudioModel(SequenceStudioModel):
    def __init__(self, decisions: list[StudioDecision]) -> None:
        super().__init__(decisions)
        self.contexts: list[dict[str, object]] = []

    async def decide(self, context: dict[str, object]) -> StudioReply:
        self.contexts.append(context)
        return await super().decide(context)


class NeverDecideModel:
    async def decide(self, _context: dict[str, object]) -> StudioReply:
        raise AssertionError("运行状态事实问题不应该交给模型猜测")


def test_context_limit_uses_selected_model() -> None:
    assert context_limit_for_model("deepseek", "deepseek-v4-flash") == 1_000_000
    assert context_limit_for_model("deepseek", "deepseek-v4-pro") == 1_000_000
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert context_limit_for_model("openai", model) == 1_050_000
    assert context_limit_for_model("openai", "unknown-proxy-model") == 16_000


def test_bulk_delete_one_approval_and_turn_local_reply(tmp_path: Path) -> None:
    from veripatch.studio_permissions import fingerprint

    root = tmp_path / "repo"
    root.mkdir()
    for name in ("app.py", "test_app.py", "test_other.py"):
        (root / name).write_text("hello", encoding="utf-8")
    session = StudioSession(
        session_id="bulk-delete",
        repo_root=str(root),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    session.observations = [
        StudioObservation(
            kind="edit",
            summary="旧默认参数修改",
            payload={"path": "app.py", "intent": "旧默认参数修改"},
        ),
        StudioObservation(
            kind="test",
            summary="old test",
            payload={"command": ["python", "-m", "pytest"], "exit_code": 0},
        ),
    ]
    store = StudioStore(tmp_path / "state.db")
    agent = StudioAgent(SequenceStudioModel([]), store)
    asyncio.run(agent.handle(session, "删除全部文件"))
    assert session.status == "waiting_permission"
    assert len(list(root.iterdir())) == 3
    decision = StudioDecision.model_validate(session.pending_permission.decision)
    assert len(decision.actions) == 3
    session.once_grants.append(fingerprint(decision))
    session.resume_decision = decision
    session.pending_permission = None
    asyncio.run(
        agent.handle(
            session,
            "删除全部文件",
            continuation=True,
            record_user_message=False,
            resume_after_permission=True,
        )
    )
    assert session.status == "completed"
    assert not list(root.iterdir())
    assert session.verification_passed is False
    reply = session.messages[-1].content
    assert "旧默认参数" not in reply
    assert "静态检查" not in reply
    assert reply.count("完成内容") == 1
    assert len([o for o in session.observations if o.kind == "delete"]) == 3


def test_command_file_effects_invalidate_old_verification(tmp_path: Path) -> None:
    from veripatch import studio_completion

    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("hello", encoding="utf-8")
    session = StudioSession(
        session_id="effects",
        repo_root=str(root),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    session.verification_passed = True
    before = studio_completion.snapshot(root, artifacts=False)
    target.unlink()
    StudioAgent._record_command_effects(
        session, SafeWorkspace(root), before, StudioStore(tmp_path / "state.db")
    )
    assert session.turn_changed_files == ["a.txt"]
    assert session.action_epoch == 1
    assert not session.verification_passed
    assert studio_completion.file_effects_verified(session)


def test_feature_goal_is_not_a_lexical_completion_gate(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="default-argument",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    message = "修改 app.py，支持不传 b，默认 b=0。"
    session.task_contract = StudioAgent._build_task_contract(session, message)
    assert "不传 b" in session.task_contract.objective
    assert not any(r.key == "feature" for r in session.task_contract.requirements)
    # Existing paused sessions must discard the obsolete gate, not mark it proven.
    session.task_contract.requirements.append(
        StudioRequirement(key="feature", description="实现功能：不传 b", expected="不传 b")
    )
    assert StudioAgent._validate_task_contract(session)  # No real change yet.
    session.turn_changed_files = ["app.py"]
    assert StudioAgent._validate_task_contract(session) == []
    assert not any(r.key == "feature" for r in session.task_contract.requirements)
    session.task_contract.requirements.append(
        StudioRequirement(key="verification", description="运行验证")
    )
    assert StudioAgent._validate_task_contract(session)  # No passing verification yet.
    session.verification_passed = True
    assert StudioAgent._validate_task_contract(session) == []


def test_long_background_does_not_become_task_objective(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="long-objective",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    # Self-contained fixture: real requests can have arbitrarily long background.
    message = "仅供参考的背景资料。" * 600 + "只创建 probe.txt，内容为 hello，不运行命令。"
    semantic = SemanticIntentAssessment(
        intent="change",
        confidence="high",
        requires_clarification=False,
        objectives=["创建 probe.txt，内容为 hello"],
        prohibited_actions=["run_command", "run_tests"],
    )
    contract = StudioAgent._build_task_contract(session, message, semantic=semantic)
    assert contract.objective == semantic.objectives[0]
    assert "背景" not in contract.objective
    fallback = StudioAgent._build_task_contract(session, message)
    assert fallback.objective == message


def test_final_marked_task_controls_requirements(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="marked-task",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    message = (
        "资料中提到实现这些功能，但只是背景。\n"
        "本轮唯一-任务：只创建 compression_probe.txt，内容为 blue-whale-729，不要运行任何命令。"
    )
    contract = StudioAgent._build_task_contract(session, message)
    descriptions = [item.description for item in contract.requirements]
    assert any("compression_probe.txt" in item for item in descriptions)
    assert not any("实现功能：这些功能" in item for item in descriptions)


def test_final_review_ignores_denied_actions_from_previous_turn(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="turn-audit",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
        observations=[StudioObservation(kind="delete", summary="旧任务删除文件")],
        turn_observation_start=1,
        task_state=StudioTaskState(denied_actions=["delete_path", "create"]),
    )
    review = StudioAgent._review_final_result(session, "已读取文件，未修改内容。")
    assert "禁止动作" not in review.reviewed_message


def test_context_budget_pause_returns_session(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="budget-pause",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    agent = StudioAgent(
        NeverDecideModel(), StudioStore(tmp_path / "state.db"), max_context_tokens=1
    )
    result = asyncio.run(agent.handle(session, "只读取并分析项目结构，不修改文件。"))
    assert result is session
    assert result.status == "paused"
    assert "\u6a21\u578b\u53ef\u53d1\u9001\u8303\u56f4" in result.pause_reason


def test_improvement_advice_is_read_only_but_explicit_improvement_is_a_change() -> None:
    request = "全面介绍一下这个项目，包括文件作用、运行方式、代码结构、优点和可以改进的地方"
    assert _requests_code_change(request) is False
    assert _requests_code_change("请改进这个项目的界面") is True
    assert _requests_code_change("根据改进建议修改代码") is True


def test_context_history_question_is_answered_from_audit_evidence(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "grounded-context.sqlite3")
    session = StudioSession(
        session_id="grounded-context",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    store.save(
        session,
        "context_trimmed",
        {
            "summary": "已裁剪可重新检索的上下文。",
            "before_tokens": 4300,
            "after_tokens": 3990,
            "limit_tokens": 4000,
            "trimmed": ["retrieved_snippets", "older_observations"],
        },
    )

    class EvidenceModel:
        async def decide(self, context):
            facts = context["audit_facts"]
            assert facts["context_compression_count"] == 0
            assert facts["latest_context_trim"]["payload"]["after_tokens"] == 3990
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="依据审计记录区分裁剪与压缩",
                    message="只有上下文裁剪记录，没有旧消息或长内容被压缩的证据。",
                )
            )

    result = asyncio.run(
        StudioAgent(EvidenceModel(), store).handle(session, "你什么时候压缩的上下文？")
    )

    assert "只有上下文裁剪记录" in result.messages[-1].content
    assert "没有旧消息或长内容被压缩的证据" in result.messages[-1].content
    assert not any(
        event["event_type"] == "fact_evidence" for event in store.events(session.session_id)
    )


def test_unified_task_state_finishes_change_when_user_forbids_commands(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("VERSION = '0.4'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_hello.py").write_text("# protected\n", encoding="utf-8")
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="只修改被授权的文件",
                path="hello.py",
                old_text="VERSION = '0.4'",
                new_text="VERSION = '0.5'",
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="已完成授权修改，按要求跳过验证",
                message="已完成修改。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "unified-task-state.sqlite3")
    session = StudioSession(
        session_id="unified-task-state",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
        approved_write_paths=[str(tmp_path)],
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "修改 hello.py，把 VERSION 改为 0.5，"
            "但不要修改 tests/test_hello.py，也不要运行任何命令。",
        )
    )

    assert result.status == "completed"
    assert result.task_state is not None
    assert result.task_state.verification_policy.value == "skipped_by_user"
    assert "verification" in result.task_state.skipped_actions
    assert StudioAction.RUN_COMMAND.value in result.task_state.denied_actions
    assert all(item.key != "verify" for item in result.plan)
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "VERSION = '0.5'\n"
    assert (tmp_path / "tests" / "test_hello.py").read_text(encoding="utf-8") == "# protected\n"
    assert not any(item.kind in {"test", "command"} for item in result.observations)
    assert "按用户要求未运行任何验证命令" in result.messages[-1].content


def test_conflicting_verification_and_command_ban_requires_clarification(
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text("VERSION = '2.0'\n", encoding="utf-8")
    store = StudioStore(tmp_path / "verification-conflict.sqlite3")
    session = StudioSession(
        session_id="verification-conflict",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
    )

    result = asyncio.run(
        StudioAgent(NeverDecideModel(), store).handle(
            session,
            "修改 hello.py，完成后运行测试，但不要运行任何命令。",
        )
    )

    assert result.status == "idle"
    assert result.activity == "waiting_user"
    assert result.task_contract is not None
    assert result.task_contract.requires_clarification is True
    assert "运行测试需要执行命令" in result.messages[-1].content
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "VERSION = '2.0'\n"


def test_verification_status_question_with_command_ban_remains_read_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="只读分析", path="hello.py"),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="如实说明验证状态",
                message="hello.py 会输出 hello；按要求未运行测试，因此无法判断测试是否通过。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "verification-status-question.sqlite3")
    session = StudioSession(
        session_id="verification-status-question",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "只读取并分析 hello.py，不要运行任何命令。最后告诉我是否通过了测试。",
        )
    )

    assert result.status == "completed"
    assert [item.kind for item in result.observations] == ["read", "result_review"]
    assert "未运行测试" in result.messages[-1].content
    assert result.task_state is not None
    assert StudioAction.RUN_TESTS.value in result.task_state.denied_actions


def test_completion_message_never_claims_verification_without_evidence(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="unverified-completion",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
        turn_changed_files=["hello.py"],
    )
    contract = StudioAgent._build_task_contract(session, "修改 hello.py，不要运行任何命令。")
    session.task_contract = contract
    session.task_state = StudioAgent._build_task_state(
        session, contract, "修改 hello.py，不要运行任何命令。"
    )

    message = StudioAgent._completion_message(session, [])

    assert "按用户要求未运行任何验证命令" in message
    assert "本地验证已通过" not in message


def test_read_only_contract_routes_model_mutation_to_explicit_approval(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "read-only-guard.sqlite3")
    session = StudioSession(
        session_id="read-only-guard",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        task_contract=StudioAgent._build_task_contract(
            StudioSession(
                session_id="policy-source",
                repo_root=str(tmp_path),
                provider="openai",
                model="gpt-test",
                reasoning_effort="low",
            ),
            "介绍项目结构、优点和可以改进的地方",
        ),
    )
    agent = StudioAgent(ModelMustNotBeCalled(), store)
    workspace = SafeWorkspace(tmp_path)

    assert (
        agent._execute(
            session,
            workspace,
            StudioDecision(
                action=StudioAction.CREATE,
                rationale="误将建议当成修改",
                path="README.md",
                content="should not exist",
            ),
        )
        is True
    )
    assert not (tmp_path / "README.md").exists()
    assert session.status == "waiting_permission"
    assert session.pending_permission is not None
    assert session.pending_permission.operation == StudioAction.CREATE.value
    assert session.pending_permission.decision is not None


def test_task_contract_persists_intent_policy_and_capability_matrix(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="intent-policy",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )

    analysis = StudioAgent._build_task_contract(session, "全面介绍项目，并说明优点和可以改进的地方")
    assert analysis.intent == "analysis"
    assert analysis.intent_confidence == "high"
    assert analysis.intent_rationale
    assert StudioAction.READ.value in analysis.allowed_actions
    assert StudioAction.CREATE.value not in analysis.allowed_actions
    assert StudioAction.RUN_COMMAND.value not in analysis.allowed_actions

    change = StudioAgent._build_task_contract(session, "请修改 app.py 并运行测试")
    assert change.intent == "change"
    assert StudioAction.EDIT.value in change.allowed_actions
    assert StudioAction.RUN_TESTS.value in change.allowed_actions


def test_contextual_continue_inherits_unfinished_contract(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "contextual-resume.sqlite3")
    session = StudioSession(
        session_id="contextual-resume",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        status="paused",
        task_contract=StudioTaskContract(
            objective="修改 app.py",
            intent="change",
            allowed_actions=[action.value for action in StudioAction],
        ),
        plan=[StudioPlanItem(key="change", title="修改", status=PlanStatus.IN_PROGRESS)],
    )
    result = asyncio.run(
        StudioAgent(
            SequenceStudioModel(
                [
                    StudioDecision(
                        action=StudioAction.RESPOND,
                        rationale="确认继承任务",
                        message="已继续原修改任务。",
                    )
                ]
            ),
            store,
            max_steps=1,
        ).handle(session, "接着做")
    )

    assert result.task_contract is not None
    assert result.task_contract.objective == "修改 app.py"
    assert result.messages[-1].content == "已继续原修改任务。"


def test_ambiguous_side_effect_request_requires_clarification(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "intent-clarification.sqlite3")
    session = StudioSession(
        session_id="intent-clarification",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        status="completed",
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), store).handle(session, "先看看，不行的话处理一下")
    )

    assert result.activity == "waiting_user"
    assert result.pause_reason == "需要确认含糊的操作意图"
    assert "仅分析、修改文件，还是运行命令" in result.messages[-1].content
    assert not result.changed_files


def test_explicit_change_authority_overrides_semantic_uncertainty(
    tmp_path: Path,
) -> None:
    class UncertainSemanticModel(SequenceStudioModel):
        def __init__(self) -> None:
            super().__init__(
                [
                    StudioDecision(
                        action=StudioAction.EDIT,
                        rationale="已按明确授权完成",
                        path="hello.py",
                        old_text="VERSION = '1.0'",
                        new_text="VERSION = '1.1'",
                    ),
                    StudioDecision(
                        action=StudioAction.FINISH,
                        rationale="已按要求跳过命令",
                        message="收到明确修改授权，未要求澄清。",
                    ),
                ]
            )
            self.classifier_calls = 0

        async def classify_intent(
            self, _messages: list[dict[str, str]], _message: str
        ) -> SemanticIntentAssessment:
            self.classifier_calls += 1
            return SemanticIntentAssessment(
                intent="analysis",
                confidence="medium",
                requires_clarification=True,
                rationale="兼容模型误判为需要澄清",
            )

    model = UncertainSemanticModel()
    (tmp_path / "hello.py").write_text("VERSION = '1.0'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_hello.py").write_text("# untouched\n", encoding="utf-8")
    session = StudioSession(
        session_id="explicit-authority",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
        approved_write_paths=[str(tmp_path)],
    )

    result = asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "explicit-authority.sqlite3")).handle(
            session,
            "修改 hello.py，把 VERSION 改为 1.1，"
            "但不要修改 tests/test_hello.py，也不要运行任何命令。",
        )
    )

    assert model.classifier_calls == 0
    assert result.activity != "waiting_user"
    assert result.task_contract is not None
    assert result.task_contract.requires_clarification is False
    assert result.task_contract.intent == "change"
    assert StudioAction.EDIT.value in result.task_contract.allowed_actions
    assert StudioAction.RUN_COMMAND.value not in result.task_contract.allowed_actions
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "VERSION = '1.1'\n"
    assert (tmp_path / "tests" / "test_hello.py").read_text(encoding="utf-8") == "# untouched\n"
    assert not any(item.kind in {"test", "command"} for item in result.observations)
    assert "请明确说明" not in result.messages[-1].content


def test_self_contained_deliberative_question_skips_semantic_classifier(tmp_path: Path) -> None:
    class SemanticModel(SequenceStudioModel):
        calls = 0

        async def classify_intent(
            self, _messages: list[dict[str, str]], _message: str
        ) -> SemanticIntentAssessment:
            self.calls += 1
            return SemanticIntentAssessment(
                intent="analysis",
                confidence="high",
                requires_clarification=False,
                rationale="用户在评估可能的调整，没有要求实施",
            )

    model = SemanticModel(
        [
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="回答设计问题",
                message="可以调整，但当前没有修改文件。",
            )
        ]
    )
    session = StudioSession(
        session_id="semantic-analysis",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    result = asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "semantic.sqlite3")).handle(
            session, "你觉得这个结构是不是应该调整？"
        )
    )

    assert model.calls == 0
    assert result.task_contract is not None
    assert result.task_contract.intent == "answer"
    assert result.task_contract.intent_source == "deterministic"
    assert StudioAction.EDIT.value not in result.task_contract.allowed_actions


def test_explicit_file_read_bypasses_redundant_semantic_classifier(tmp_path: Path) -> None:
    class PrimarySemanticModel(SequenceStudioModel):
        calls = 0

        async def classify_intent(
            self, _messages: list[dict[str, str]], _message: str
        ) -> SemanticIntentAssessment:
            self.calls += 1
            return SemanticIntentAssessment(
                intent="analysis",
                confidence="high",
                requires_clarification=False,
                rationale="用户要求只读分析文件并报告验证状态",
                dialogue_act="analyze_and_report_status",
                objectives=["分析 hello.py", "报告现有验证状态"],
                questions=["是否存在测试通过证据"],
                requested_actions=["read", "respond"],
                prohibited_actions=["run_command", "run_tests", "edit"],
                references=["hello.py"],
            )

    model = PrimarySemanticModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="读取", path="hello.py"),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="回答",
                message="已分析；没有运行测试。",
            ),
        ]
    )
    (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    session = StudioSession(
        session_id="primary-semantic",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
    )

    result = asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "primary-semantic.sqlite3")).handle(
            session, "只读取并分析 hello.py，不要运行任何命令，告诉我测试状态"
        )
    )

    assert model.calls == 0
    assert result.task_contract is not None
    assert result.task_contract.intent == "analysis"
    assert StudioAction.READ.value in result.task_contract.allowed_actions
    assert StudioAction.RUN_COMMAND.value in result.task_contract.denied_actions


def test_semantic_classifier_accepts_numeric_confidence_from_compatible_models() -> None:
    assessment = SemanticIntentAssessment.model_validate(
        {
            "intent": "analysis",
            "confidence": 0.96,
            "requires_clarification": False,
            "rationale": "用户要求只读分析",
        }
    )

    assert assessment.confidence == "high"


def test_user_message_is_persisted_before_semantic_classifier_finishes(
    tmp_path: Path,
) -> None:
    class SlowSemanticModel(SequenceStudioModel):
        def __init__(self) -> None:
            super().__init__(
                [
                    StudioDecision(
                        action=StudioAction.FINISH,
                        rationale="直接回答",
                        message="分析完成。",
                    )
                ]
            )
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def classify_intent(
            self, _messages: list[dict[str, str]], _message: str
        ) -> SemanticIntentAssessment:
            self.started.set()
            await self.release.wait()
            return SemanticIntentAssessment(
                intent="analysis",
                confidence="high",
                requires_clarification=False,
                rationale="用户要求分析",
            )

    async def scenario() -> StudioSession:
        model = SlowSemanticModel()
        store = StudioStore(tmp_path / "early-message.sqlite3")
        session = StudioSession(
            session_id="early-message",
            repo_root=str(tmp_path),
            provider="deepseek",
            model="deepseek-v4-pro",
            reasoning_effort="low",
            messages=[
                StudioMessage(role="user", content="看一下项目的分层"),
                StudioMessage(role="assistant", content="可以从依赖方向分析。"),
            ],
        )
        task = asyncio.create_task(StudioAgent(model, store).handle(session, "分析这个项目结构"))
        await model.started.wait()
        persisted = store.load(session.session_id)
        assert persisted is not None
        assert persisted.status == "running"
        assert persisted.messages[-1].role == "user"
        assert persisted.messages[-1].content == "分析这个项目结构"
        model.release.set()
        return await task

    result = asyncio.run(scenario())

    assert [item.content for item in result.messages].count("分析这个项目结构") == 1


def test_semantic_classifier_cannot_self_authorize_hidden_change(tmp_path: Path) -> None:
    class SemanticMutationModel:
        async def classify_intent(
            self, _messages: list[dict[str, str]], _message: str
        ) -> SemanticIntentAssessment:
            return SemanticIntentAssessment(
                intent="change",
                confidence="high",
                requires_clarification=False,
                rationale="可能暗示希望修改",
            )

        async def decide(self, _context: dict[str, object]) -> StudioReply:
            raise AssertionError("clarification must happen before the action model")

    session = StudioSession(
        session_id="semantic-no-escalation",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        messages=[
            StudioMessage(role="user", content="我们刚才在讨论模块边界"),
            StudioMessage(role="assistant", content="目前只做了分析。"),
        ],
    )
    result = asyncio.run(
        StudioAgent(
            SemanticMutationModel(), StudioStore(tmp_path / "no-escalation.sqlite3")
        ).handle(session, "你觉得这个结构是不是应该调整？")
    )

    assert result.activity == "waiting_user"
    assert result.task_contract is not None
    assert result.task_contract.requires_clarification is True
    assert StudioAction.EDIT.value not in result.task_contract.allowed_actions


def test_semantic_classifier_failure_falls_back_to_safe_local_policy(tmp_path: Path) -> None:
    class FailingSemanticModel(SequenceStudioModel):
        async def classify_intent(
            self, _messages: list[dict[str, str]], _message: str
        ) -> SemanticIntentAssessment:
            raise RuntimeError("classifier unavailable")

    model = FailingSemanticModel(
        [
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="安全回答",
                message="这里只讨论设计，不修改文件。",
            )
        ]
    )
    store = StudioStore(tmp_path / "semantic-fallback.sqlite3")
    session = StudioSession(
        session_id="semantic-fallback",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        messages=[
            StudioMessage(role="user", content="先评估一下项目分层"),
            StudioMessage(role="assistant", content="已保留评估上下文。"),
        ],
    )
    result = asyncio.run(
        StudioAgent(model, store).handle(session, "你觉得这个结构是不是应该调整？")
    )

    assert result.task_contract is not None
    assert StudioAction.EDIT.value not in result.task_contract.allowed_actions
    assert any(
        event["event_type"] == "intent_classifier_fallback"
        for event in store.events(session.session_id)
    )


def test_long_conversation_builds_durable_structured_summary(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="long-summary",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        task_contract=StudioTaskContract(objective="修复登录", intent="change"),
        messages=[
            StudioMessage(role="user" if index % 2 == 0 else "assistant", content=f"消息 {index}")
            for index in range(40)
        ],
    )
    session.memory.constraints = ["不要修改前端"]
    session.memory.relevant_files = ["src/auth.py"]
    session.observations.append(
        StudioObservation(kind="edit", summary="已修改 src/auth.py。", payload={})
    )
    session.plan = [StudioPlanItem(key="verify", title="运行验证", status=PlanStatus.PENDING)]

    StudioAgent._refresh_context_summary(session)

    assert session.context_summary.objective == "修复登录"
    assert session.context_summary.summarized_message_count == 28
    assert session.context_summary.constraints == ["不要修改前端"]
    assert session.context_summary.completed_actions == ["已修改 src/auth.py。"]
    assert session.context_summary.pending_steps == ["运行验证"]


def test_followup_correction_revokes_previous_write_authority(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="correction",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        task_contract=StudioAgent._build_task_contract(
            StudioSession(
                session_id="old",
                repo_root=str(tmp_path),
                provider="openai",
                model="gpt-test",
                reasoning_effort="low",
            ),
            "修改登录模块",
        ),
    )
    policy = classify_intent("不是让你改，只分析原因")
    updated = StudioAgent._updated_task_contract(session, "不是让你改，只分析原因", policy)

    assert updated.intent in {"answer", "analysis"}
    assert StudioAction.EDIT.value not in updated.allowed_actions
    assert updated.intent_source == "followup_update"


def test_conditional_change_authorizes_verification_before_mutation(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="conditional",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    message = "如果测试失败，再修改实现"
    updated = StudioAgent._updated_task_contract(session, message, classify_intent(message))

    assert updated.intent == "verify"
    assert StudioAction.RUN_TESTS.value in updated.allowed_actions
    assert StudioAction.EDIT.value not in updated.allowed_actions
    assert "条件成立" in str(updated.intent_rationale)


def test_runtime_steer_discards_pending_model_action_and_rebuilds_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    queued = [None, "不是让你改，只分析原因"]

    def consume() -> str | None:
        return queued.pop(0) if queued else None

    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="旧计划准备修改",
                path="app.py",
                old_text="value = 1",
                new_text="value = 2",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="按纠正只读回答",
                message="已停止修改，只分析原因。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "runtime-steer.sqlite3")
    session = StudioSession(
        session_id="runtime-steer",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    result = asyncio.run(
        StudioAgent(model, store, consume_steer=consume).handle(
            session, "修改 app.py，将 value 改为 2"
        )
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert result.task_contract is not None
    assert result.task_contract.intent in {"answer", "analysis"}
    assert StudioAction.EDIT.value not in result.task_contract.allowed_actions
    assert any(event["event_type"] == "steer_applied" for event in store.events(session.session_id))


def test_runtime_steer_reports_files_changed_before_correction(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    queued = [None, None, "不是让你改，停止修改，只分析这个文件的作用"]

    def consume() -> str | None:
        return queued.pop(0) if queued else None

    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="先执行原任务修改",
                path="app.py",
                old_text="value = 1",
                new_text="value = 2",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="按纠正只读回答",
                message="app.py 用于定义应用数值。纠正后未再修改。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "runtime-steer-after-edit.sqlite3")
    session = StudioSession(
        session_id="runtime-steer-after-edit",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        approved_write_paths=[str(tmp_path)],
    )
    result = asyncio.run(
        StudioAgent(model, store, consume_steer=consume).handle(
            session, "修改 app.py，将 value 改为 2"
        )
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 2\n"
    final_message = result.messages[-1].content
    assert "在收到纠正前，已经修改了 app.py" in final_message
    assert "收到纠正后已停止旧计划" in final_message
    steer_event = next(
        event
        for event in store.events(session.session_id)
        if event["event_type"] == "steer_applied"
    )
    assert steer_event["payload"]["prior_changed_files"] == ["app.py"]


class RecordingStudioModel(SequenceStudioModel):
    def __init__(self, decisions: list[StudioDecision]) -> None:
        super().__init__(decisions)
        self.contexts: list[dict[str, object]] = []

    async def decide(self, context: dict[str, object]) -> StudioReply:
        self.contexts.append(context)
        return await super().decide(context)


def test_model_protocol_metadata_is_persisted_in_trace(tmp_path: Path) -> None:
    class ProtocolModel:
        async def decide(self, context: dict[str, object]) -> StudioReply:
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="直接回答",
                    message="完成。",
                ),
                model="gpt-test",
                protocol="chat-json",
                fallback_reason="chat-tools-http-422",
                response_id="req-42",
                latency_ms=321,
                status_code=200,
            )

    store = StudioStore(tmp_path / "protocol.sqlite3")
    session = StudioSession(
        session_id="protocol",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    asyncio.run(StudioAgent(ProtocolModel(), store).handle(session, "你是谁"))

    event = next(
        item for item in store.events(session.session_id) if item["event_type"] == "model_response"
    )
    assert event["payload"]["protocol"] == "chat-json"
    assert event["payload"]["fallback"] is True
    assert event["payload"]["fallback_reason"] == "chat-tools-http-422"
    assert event["payload"]["response_id"] == "req-42"
    assert event["payload"]["latency_ms"] == 321
    assert event["payload"]["tool_names"] == ["respond"]
    assert event["payload"]["tool_targets"] == ["无额外目标"]


def test_context_actual_input_uses_provider_usage_not_local_estimate(tmp_path: Path) -> None:
    class UsageModel:
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, context: dict[str, object]) -> StudioReply:
            self.calls += 1
            return StudioReply(
                decision=StudioDecision(action="respond", rationale="answer", message="好的。"),
                input_tokens=321 if self.calls == 1 else 0,
            )

    store = StudioStore(tmp_path / "usage.sqlite3")
    session = StudioSession(
        session_id="usage",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    agent = StudioAgent(UsageModel(), store)

    first = asyncio.run(agent.handle(session, "你好"))
    assert first.context_estimated_tokens > 0
    assert first.context_actual_input_tokens == 321
    assert store.load(session.session_id).context_actual_input_tokens == 321

    second = asyncio.run(agent.handle(session, "再回答一次"))
    assert second.context_actual_input_tokens is None
    responses = [
        event for event in store.events(session.session_id)
        if event["event_type"] == "model_response"
    ]
    assert [event["payload"]["input_tokens"] for event in responses] == [321, None]


def test_latest_task_message_skips_permission_adjustment() -> None:
    session = StudioSession(
        session_id="latest-task",
        repo_root=".",
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        messages=[
            StudioMessage(role="user", content="把 Go 那个启动一下"),
            StudioMessage(role="user", content="调整执行方案：安装到 D 盘"),
        ],
    )

    assert _latest_task_message(session) == "把 Go 那个启动一下"


def test_pure_bulk_delete_request_excludes_follow_up_work() -> None:
    assert _is_pure_bulk_delete_request("删除这个文件夹里的全部东西") is True
    assert _is_pure_bulk_delete_request("清空所有文件，然后创建 README") is False
    assert _is_pure_bulk_delete_request("delete everything in this folder") is True


def test_studio_store_returns_latest_event_window_by_default(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "event-window.sqlite3")
    session = StudioSession(
        session_id="event-window",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    for index in range(510):
        store.save(session, "tick", {"index": index})

    events = store.events(session.session_id)

    assert len(events) == 500
    assert events[0]["payload"]["index"] == 10
    assert events[-1]["payload"]["index"] == 509


class VerifiedThenDisconnectedModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, context: dict[str, object]) -> StudioReply:
        self.calls += 1
        if self.calls == 1:
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.CREATE,
                    rationale="创建可验证脚本",
                    path="hello.py",
                    content='print("ok")\n',
                )
            )
        if self.calls == 2:
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RUN_COMMAND,
                    rationale="运行脚本验证",
                    command=["python", "hello.py"],
                )
            )
        raise RuntimeError("upstream disconnected during final summary")


class ModelMustNotBeCalled:
    async def decide(self, context: dict[str, object]) -> StudioReply:
        raise AssertionError("explicit launch intent must bypass the model")


class AlwaysTimeoutModel:
    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    async def decide(self, context: dict[str, object]) -> StudioReply:
        self.contexts.append(context)
        raise TimeoutError("upstream stalled")


def test_store_repairs_oversized_legacy_permission_history(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="legacy",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    payload = session.model_dump(mode="json")
    payload["approved_commands"] = [["cmd", "/c", f"echo-{index}"] for index in range(41)]

    recovered = StudioStore._decode_session(json.dumps(payload))

    assert len(recovered.approved_commands) == 40
    assert recovered.approved_commands[0][-1] == "echo-1"


class SlowManagedModel:
    manages_request_timeout = True

    async def decide(self, context: dict[str, object]) -> StudioReply:
        await asyncio.sleep(0.02)
        return StudioReply(
            decision=StudioDecision(
                action=StudioAction.RESPOND,
                rationale="流仍然活跃",
                message="完成",
            )
        )


def test_timeout_retries_with_compact_context_then_pauses_resumably(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    model = AlwaysTimeoutModel()
    store = StudioStore(tmp_path / "timeout.sqlite3")
    session = StudioSession(
        session_id="timeout",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        messages=[StudioMessage(role="user", content="old " + "x" * 4_000)],
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "修复任务"))

    assert len(model.contexts) == 2
    assert result.status == "paused"
    assert result.activity == "paused"
    assert result.failure_reason is None
    assert result.pause_reason is not None
    assert StudioAgent._estimate_tokens(model.contexts[1]) < StudioAgent._estimate_tokens(
        model.contexts[0]
    )
    assert "recent_observations" not in model.contexts[1]
    assert all("payload" not in item for item in model.contexts[1].get("recent_evidence", []))
    event_types = [event["event_type"] for event in store.events("timeout")]
    assert "model_retrying" in event_types
    assert "model_diagnostic" in event_types
    assert event_types[-1] == "paused"
    assert result.messages[-1].content.startswith("调用诊断\n\n")
    assert "中转站、网络代理或上游连接" in result.messages[-1].content


def test_provider_managed_stream_is_not_cut_off_by_agent_wall_clock(tmp_path: Path) -> None:
    agent = StudioAgent(SlowManagedModel(), StudioStore(tmp_path / "managed.sqlite3"))

    reply = asyncio.run(agent._await_model_decision({}, fallback_timeout=0.001))

    assert reply.decision.message == "完成"


def test_verified_changes_complete_automatically_when_turn_budget_ends(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="修复实现",
                path="calc.py",
                old_text="return a - b",
                new_text="return a + b",
            ),
            StudioDecision(
                action=StudioAction.RUN_TESTS,
                rationale="验证实现",
                command=["python", "-m", "pytest", "-q"],
            ),
        ]
    )
    store = StudioStore(tmp_path / "budget-complete.sqlite3")
    session = StudioSession(
        session_id="budget-complete",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        response_style=ResponseStyle.STANDARD,
    )

    result = asyncio.run(StudioAgent(model, store, max_steps=2).handle(session, "修复加法"))

    assert result.status == "completed"
    assert result.verification_passed is True
    assert result.review_completed is True
    assert "所执行的测试已通过" in result.messages[-1].content
    assert "python -m pytest" not in result.messages[-1].content
    assert "完成内容\n- 已修改 calc.py" in result.messages[-1].content
    assert result.messages[-1].content.count("完成内容") == 1
    assert "改动规模\n- 新增 1 行 · 删除 1 行" in result.messages[-1].content


def test_resume_restores_successful_approved_validation_evidence(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    store = StudioStore(tmp_path / "evidence-resume.sqlite3")
    session = StudioSession(
        session_id="evidence-resume",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="paused",
        changed_files=["calc.py"],
        turn_changed_files=["calc.py"],
        plan=StudioAgent._build_plan(VerificationMode.AUTO),
        observations=[
            StudioObservation(
                kind="edit",
                summary="已修改",
                payload={
                    "path": "calc.py",
                    "diff": "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-return a-b\n+return a+b\n",
                },
            ),
            StudioObservation(
                kind="command",
                summary="验证通过",
                payload={
                    "command": ["python", "-c", "from calc import add; assert add(2, 3) == 5"],
                    "exit_code": 0,
                },
            ),
        ],
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), store).handle(session, "继续", continuation=True)
    )

    assert result.status == "completed"
    assert result.verification_passed is True


def test_redundant_read_from_previous_turn_can_be_refreshed_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    model = RecordingStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="再读一次", path="index.html"),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="使用已有证据继续",
                message="已转向下一步。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "duplicate.sqlite3")
    session = StudioSession(
        session_id="duplicate-read",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        observations=[
            StudioObservation(
                kind="read",
                summary="已读取 index.html。",
                payload={"path": "index.html", "content": "<html></html>"},
            )
        ],
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "检查最新内容"))

    assert result.status == "idle"
    assert len([item for item in result.observations if item.kind == "read"]) == 2
    assert not any(item.kind == "duplicate_action" for item in result.observations)


def test_continue_reuses_cross_turn_list_files_instead_of_running_it_again(
    tmp_path: Path,
) -> None:
    store = StudioStore(tmp_path / "cross-turn-ledger.sqlite3")
    session = StudioSession(
        session_id="cross-turn-ledger",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    first_model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.LIST_FILES, rationale="查看目录"),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="等待下一轮",
                message="已查看。",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="仍未产生修改",
                message="已查看。",
            ),
        ]
    )
    first = asyncio.run(StudioAgent(first_model, store).handle(session, "修复项目并等待下一步"))

    resumed_model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.LIST_FILES, rationale="重复查看目录"),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="复用目录证据",
                message="已从现有证据继续。",
            ),
        ]
    )
    result = asyncio.run(StudioAgent(resumed_model, store).handle(first, "继续"))

    assert len([item for item in result.observations if item.kind == "files"]) == 1
    duplicate = next(
        item for item in reversed(result.observations) if item.kind == "duplicate_action"
    )
    assert duplicate.payload["cross_turn"] is True
    assert "Reuse the existing file evidence" in duplicate.payload["required_next_step"]
    assert result.status == "paused"
    assert result.messages[-1].content != "已从现有证据继续。"


def test_successful_mcp_read_is_available_before_any_duplicate(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="mcp-answer-phase",
        repo_root=str(tmp_path),
        provider="openai_official",
        model="gpt-6-luna",
        reasoning_effort="medium",
        task_contract=StudioTaskContract(objective="用 MCP 查看项目文件", intent="analysis"),
        observations=[
            StudioObservation(
                kind="mcp_tool",
                summary="MCP 工具 list_project_files 已调用。",
                payload={"tool": "list_project_files", "arguments": {}, "result": {"files": ["app.py"]}},
            ),
        ],
    )
    assert StudioAgent._reusable_read_evidence(session)["payload"]["result"] == {
        "files": ["app.py"]
    }
    assert StudioAgent._latest_tool_result(session)["observation_id"] == 0
    assert "足够就直接回答" in StudioAgent._next_instruction(session)


def test_explicit_mcp_request_routes_equivalent_native_read_to_mcp() -> None:
    routed = StudioAgent._route_tool_preference(
        StudioDecision(action=StudioAction.LIST_FILES, rationale="查看项目"),
        "请用 MCP 查看当前项目文件",
    )
    assert routed.action is StudioAction.MCP_CALL
    assert routed.mcp_tool == "list_project_files"
    assert StudioAgent._route_tool_preference(
        StudioDecision(action=StudioAction.LIST_FILES, rationale="查看项目"), "查看项目文件"
    ).action is StudioAction.LIST_FILES


def test_explicit_tool_requirement_blocks_premature_answer_then_accepts_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    model = SequenceStudioModel([
        StudioDecision(action=StudioAction.RESPOND, rationale="直接回答", message="可能有 app.py"),
        StudioDecision(action=StudioAction.LIST_FILES, rationale="查看本轮文件"),
        StudioDecision(action=StudioAction.RESPOND, rationale="依据工具结果回答", message="有 app.py。"),
    ])
    session = StudioSession(
        session_id="explicit-tool-evidence", repo_root=str(tmp_path), provider="openai",
        model="gpt-test", reasoning_effort="low",
    )
    result = asyncio.run(StudioAgent(
        model, StudioStore(tmp_path / "tool-evidence.sqlite3")
    ).handle(session, "请用 MCP 查看当前项目文件"))
    assert result.status == "idle"
    assert result.messages[-1].content == "有 app.py。"
    assert any(item.kind == "requirement_gate" for item in result.observations)
    assert [item.payload["tool"] for item in result.observations if item.kind == "mcp_tool"] == [
        "list_project_files"
    ]
    assert not any(item.kind == "files" for item in result.observations)


def test_named_native_tool_requirement_uses_same_answer_gate(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("ok\n", encoding="utf-8")
    model = SequenceStudioModel([
        StudioDecision(action=StudioAction.RESPOND, rationale="过早回答", message="可能有文件。"),
        StudioDecision(action=StudioAction.LIST_FILES, rationale="按要求查看"),
        StudioDecision(action=StudioAction.RESPOND, rationale="引用本轮结果", message="有 app.py。"),
    ])
    session = StudioSession(
        session_id="native-tool-evidence", repo_root=str(tmp_path), provider="openai",
        model="gpt-test", reasoning_effort="low",
    )
    result = asyncio.run(StudioAgent(
        model, StudioStore(tmp_path / "native-evidence.sqlite3")
    ).handle(session, "请用 list_files 查看项目文件"))
    assert result.status == "idle"
    assert result.messages[-1].content == "有 app.py。"
    assert any(item.kind == "requirement_gate" for item in result.observations)
    assert any(item.kind == "files" for item in result.observations)
    assert not any(item.kind == "mcp_tool" for item in result.observations)


def test_file_read_requirement_needs_current_turn_evidence(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="fresh-file-contract", repo_root=str(tmp_path), provider="openai",
        model="gpt-test", reasoning_effort="low",
        task_contract=StudioTaskContract(
            objective="读取 app.py", intent="analysis",
            requirements=[StudioRequirement(
                key="target_file", expected="app.py", description="读取 app.py",
            )],
        ),
        observations=[StudioObservation(kind="read", summary="旧轮读取", payload={"path": "app.py"})],
        turn_observation_start=1,
    )
    assert StudioAgent._validate_task_contract(session) == ["读取 app.py"]
    session.observations.append(StudioObservation(
        kind="read", summary="本轮读取", payload={"path": "app.py"},
    ))
    assert StudioAgent._validate_task_contract(session) == []


def test_controller_exposes_successful_mcp_result_without_duplicate(tmp_path: Path) -> None:
    class AnswerPhaseModel:
        async def decide(self, context: dict[str, object]) -> StudioReply:
            assert context["available_actions"] is None
            assert "app.py" in str(context["answer_evidence"])
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="复用已取得的 MCP 文件列表",
                    message="项目包含 app.py。",
                ),
                input_tokens=10,
                output_tokens=3,
            )

    session = StudioSession(
        session_id="mcp-answer-routing",
        repo_root=str(tmp_path),
        provider="openai_official",
        model="gpt-6-luna",
        reasoning_effort="medium",
        status="paused",
        task_contract=StudioTaskContract(objective="用 MCP 查看项目文件", intent="analysis"),
        observations=[
            StudioObservation(
                kind="mcp_tool",
                summary="MCP 工具 list_project_files 已调用。",
                payload={"tool": "list_project_files", "arguments": {}, "result": {"files": ["app.py"]}},
            ),
        ],
    )
    result = asyncio.run(StudioAgent(AnswerPhaseModel(), StudioStore(tmp_path / "mcp-answer.sqlite3")).handle(
        session, "继续", continuation=True, record_user_message=False
    ))
    assert result.status == "idle"
    assert result.messages[-1].content == "项目包含 app.py。"


def test_continue_turns_repeated_tool_failure_into_alternative_strategy(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "failure-strategy.sqlite3")
    session = StudioSession(
        session_id="failure-strategy",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    first_model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="读取目标", path="missing.py"),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="报告当前状态",
                message="目标文件不存在。",
            ),
        ]
    )
    first = asyncio.run(StudioAgent(first_model, store).handle(session, "修复 missing.py 的引用"))

    resumed_model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="原样重试", path="missing.py"),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="采用替代策略",
                message="将改为搜索真实路径。",
            ),
        ]
    )
    result = asyncio.run(StudioAgent(resumed_model, store).handle(first, "继续"))

    assert len([item for item in result.observations if item.kind == "tool_error"]) == 1
    duplicate = next(
        item for item in reversed(result.observations) if item.kind == "duplicate_action"
    )
    assert "确认真实路径后再执行" in duplicate.payload["required_next_step"]
    assert result.messages[-1].content == "将改为搜索真实路径。"


def test_read_only_answer_is_not_misclassified_and_continue_does_not_restart(
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text('print("hello")\n', encoding="utf-8")
    store = StudioStore(tmp_path / "read-only-complete.sqlite3")
    session = StudioSession(
        session_id="read-only-complete",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )
    model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="读取文件", path="hello.py"),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="解释已有代码",
                message="hello.py 会输出 hello，未修改文件。",
            ),
        ]
    )

    answered = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "读取 hello.py，说明它做了什么，不要修改。",
        )
    )

    assert answered.task_contract is not None
    assert answered.task_contract.intent == "analysis"
    assert answered.turn_changed_files == []
    assert StudioAgent._validate_task_contract(answered) == []

    completed = asyncio.run(StudioAgent(ModelMustNotBeCalled(), store).handle(answered, "继续"))

    assert completed.status == "completed"
    assert completed.activity == "completed"
    assert "已经完成" in completed.messages[-1].content
    assert not any(item.kind == "requirement_gate" for item in completed.observations)


def test_premature_read_only_response_is_converted_to_target_read(tmp_path: Path) -> None:
    assert StudioAgent._requests_explicit_file_read(
        "读取 hello.py，不修改文件，也不要运行命令。"
    )
    assert (
        classify_intent("读取 hello.py，不修改文件，也不要运行命令。").intent
        == "analysis"
    )
    (tmp_path / "hello.py").write_text('print("hello")\n', encoding="utf-8")
    session = StudioSession(
        session_id="read-before-answer",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="直接回答",
                message="hello.py 会输出 hello。",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="根据读取结果回答",
                message="hello.py 会输出 hello。",
            ),
        ]
    )

    result = asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "read-before-answer.sqlite3")).handle(
            session, "读取 hello.py，说明它做了什么，不要修改。"
        )
    )

    assert result.status == "idle"
    assert any(item.kind == "read" for item in result.observations)
    assert StudioAgent._validate_task_contract(result) == []


def test_premature_idempotent_response_reads_before_accepting_no_change(tmp_path: Path) -> None:
    source = "def add(a, b=0):\n    return a + b\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    session = StudioSession(
        session_id="idempotent-read",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="low",
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="现状已经满足",
                message="app.py 已满足要求。",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="根据读取证据确认",
                message="app.py 已满足要求，无需重复修改。",
            ),
        ]
    )

    result = asyncio.run(
        StudioAgent(model, StudioStore(tmp_path / "idempotent-read.sqlite3")).handle(
            session,
            "把 app.py 的 add 改成支持省略 b，默认 b=0；已经满足则不要重复修改。",
        )
    )

    assert result.status == "idle"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == source
    assert any(item.kind == "read" for item in result.observations)
    assert StudioAgent._validate_task_contract(result) == []


def test_missing_answer_target_is_satisfied_by_not_found_evidence(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="missing-answer",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        task_contract=StudioTaskContract(
            objective="读取 missing.py；不存在则查找相近文件",
            intent="answer",
            requirements=[
                StudioRequirement(
                    key="target_file",
                    description="按要求处理指定文件 missing.py",
                    expected="missing.py",
                )
            ],
        ),
        observations=[
            StudioObservation(
                kind="tool_error",
                summary="工具执行失败：FileNotFoundError: missing.py",
                payload={"action": StudioAction.READ, "path": "missing.py"},
            ),
            StudioObservation(
                kind="search",
                summary="找到 0 个匹配项。",
                payload={"query": "**/*miss*.py", "results": []},
            ),
        ],
    )

    assert StudioAgent._validate_task_contract(session) == []
    assert session.task_contract is not None
    assert session.task_contract.requirements[0].evidence == "已确认 missing.py 不存在"


def test_missing_analysis_target_is_satisfied_by_complete_file_listing(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="missing-analysis-listing",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
        task_contract=StudioTaskContract(
            objective="分析 user_service.py；不存在则查找真正对应的文件",
            intent="analysis",
            requirements=[
                StudioRequirement(
                    key="target_file",
                    description="按要求处理指定文件 user_service.py",
                    expected="user_service.py",
                )
            ],
        ),
        observations=[
            StudioObservation(
                kind="search",
                summary="找到 0 个匹配项。",
                payload={"query": "login", "results": []},
            ),
            StudioObservation(
                kind="files",
                summary="发现 3 个文件。",
                payload={"files": ["hello.py", "README.md", "tests/test_hello.py"]},
            ),
        ],
    )

    assert StudioAgent._validate_task_contract(session) == []
    assert session.task_contract is not None
    assert session.task_contract.requirements[0].evidence == ("已确认 user_service.py 不存在")


def test_nested_decision_json_is_not_rendered_as_assistant_prose() -> None:
    nested = json.dumps(
        {
            "action": "finish",
            "rationale": "已有读取证据",
            "message": "hello.py 会输出 hello，文件未修改。",
        },
        ensure_ascii=False,
    )

    assert StudioAgent._unwrap_decision_message(nested) == "hello.py 会输出 hello，文件未修改。"


def test_explicit_python_syntax_check_runs_once_and_reports_result(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("answer = 42\n", encoding="utf-8")
    store = StudioStore(tmp_path / "syntax-check.sqlite3")
    session = StudioSession(
        session_id="syntax-check",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), store).handle(
            session, "检查 calculator.py 的 Python 语法，不要修改文件。"
        )
    )

    commands = [item for item in result.observations if item.kind == "command"]
    assert len(commands) == 1
    assert commands[0].payload["command"] == ["python", "-m", "py_compile", "calculator.py"]
    assert result.status == "idle"
    assert "语法检查通过" in result.messages[-1].content
    assert not any(item.kind == "duplicate_action" for item in result.observations)


def test_continue_resumes_existing_plan_instead_of_restarting(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "resume-plan.sqlite3")
    session = StudioSession(
        session_id="resume-plan",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="failed",
        failure_reason="检测到重复动作：read",
        plan=[
            StudioPlanItem(key="understand", title="理解", status=PlanStatus.COMPLETED),
            StudioPlanItem(key="implement", title="实现", status=PlanStatus.IN_PROGRESS),
        ],
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="从实现阶段继续",
                message="已恢复。",
            )
        ]
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "继续"))

    assert [item.key for item in result.plan] == ["understand", "implement"]
    assert result.plan[0].status is PlanStatus.COMPLETED
    assert any(event["event_type"] == "plan_resumed" for event in store.events("resume-plan"))


def test_plan_size_and_steps_adapt_to_the_task() -> None:
    answer_plan = StudioAgent._build_plan(
        VerificationMode.AUTO,
        StudioTaskContract(objective="解释这个项目", intent="answer"),
        "解释这个项目",
    )
    complex_plan = StudioAgent._build_plan(
        VerificationMode.AUTO,
        StudioTaskContract(
            objective="重构多模块数据库配置并安装依赖，验证后打开",
            intent="change",
            requirements=[
                StudioRequirement(key="launch_after_change", description="验证后打开新产物")
            ],
        ),
        "重构多模块数据库配置并安装依赖，验证后打开",
    )

    assert [item.key for item in answer_plan] == ["understand", "review"]
    assert len(complex_plan) > 5
    assert {"investigate", "design", "dependencies", "launch"}.issubset(
        {item.key for item in complex_plan}
    )


def test_bare_continue_does_not_repeat_an_already_completed_task(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "completed.sqlite3")
    session = StudioSession(
        session_id="already-complete",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="completed",
        activity="completed",
        changed_files=["calculator.py"],
        plan=StudioAgent._build_plan(VerificationMode.AUTO),
    )

    result = asyncio.run(StudioAgent(ModelMustNotBeCalled(), store).handle(session, "继续"))

    assert result.status == "completed"
    assert result.messages[-1].role == "assistant"
    assert "已经完成并通过验证" in result.messages[-1].content
    assert "新的目标" in result.messages[-1].content


def test_explicit_open_request_routes_to_audited_launch_permission(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    store = StudioStore(tmp_path / "launch-intent.sqlite3")
    session = StudioSession(
        session_id="launch-intent",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["index.html"],
    )

    result = asyncio.run(StudioAgent(ModelMustNotBeCalled(), store).handle(session, "帮我打开"))

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.access == "execute"
    assert result.pending_permission.command == ["cmd", "/c", "start", "", "index.html"]
    assert result.pending_permission.capability == "launch:index.html"


def test_compound_rewrite_then_open_request_is_not_truncated_to_launch(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("print('old')\n", encoding="utf-8")
    store = StudioStore(tmp_path / "compound-launch.sqlite3")
    session = StudioSession(
        session_id="compound-launch",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py"],
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action="respond",
                rationale="需要先选择目标语言并实施重写",
                message="我会先完成重写，再打开新程序。",
            ),
            StudioDecision(
                action="respond",
                rationale="仍未实施重写",
                message="我会先完成重写，再打开新程序。",
            ),
        ]
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(session, "先换种语言写计算器，要有界面，然后打开")
    )

    assert result.pending_permission is None
    assert result.status == "paused"
    assert result.messages[-1].content != "我会先完成重写，再打开新程序。"


@pytest.mark.parametrize(
    "message",
    [
        "写个 Go 的吧，然后打开",
        "用 Go 做一个计算器再打开",
        "生成一个 Go 图形计算器并打开",
        "把它转换成 Go 后打开",
        "把go的计算机改的高级点，然后打开",
        "把 Go 计算器改得更美观再启动",
    ],
)
def test_colloquial_compound_build_requests_do_not_open_the_old_file(
    tmp_path: Path, message: str
) -> None:
    (tmp_path / "calculator.py").write_text("print('old')\n", encoding="utf-8")
    store = StudioStore(tmp_path / "colloquial-compound.sqlite3")
    session = StudioSession(
        session_id="colloquial-compound",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py"],
    )
    model = SequenceStudioModel([
        StudioDecision(action="respond", rationale="先实现 Go 版本", message="开始实现。"),
        StudioDecision(action="respond", rationale="仍未实现 Go 版本", message="开始实现。"),
    ])

    result = asyncio.run(StudioAgent(model, store).handle(session, message))

    assert result.pending_permission is None
    assert result.status == "paused"
    assert result.messages[-1].content != "开始实现。"
    assert result.task_contract is not None
    assert result.task_contract.intent == "change"
    assert any(
        item.key == "target_language" and item.expected == ".go"
        for item in result.task_contract.requirements
    )
    assert any(item.key == "launch_after_change" for item in result.task_contract.requirements)
    assert any(item.key == "workspace_change" for item in result.task_contract.requirements)


def test_automatic_verifier_matches_changed_project_stack(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("pass")
    (tmp_path / "helper.py").write_text("pass")
    python_session = StudioSession(
        session_id="python-verifier",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        turn_changed_files=["app.py", "helper.py"],
    )
    python = StudioAgent._automatic_verification_decision(python_session)
    assert python is not None
    assert python.command == ["python", "-m", "py_compile", "app.py", "helper.py"]

    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    go_session = python_session.model_copy(
        update={"session_id": "go-verifier", "turn_changed_files": ["main.go"]}
    )
    go = StudioAgent._automatic_verification_decision(go_session)
    assert go is not None
    assert go.command == ["go", "test", "./..."]

    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"build": "vite build"}}), encoding="utf-8"
    )
    node_session = python_session.model_copy(
        update={"session_id": "node-verifier", "turn_changed_files": ["src/app.ts"]}
    )
    node = StudioAgent._automatic_verification_decision(node_session)
    assert node is not None
    assert node.command == ["npm", "run", "build"]


def test_task_contract_tracks_target_feature_protection_and_preservation(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="rich-contract",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    message = (
        "修改 calculator.py，增加一个复制结果按钮，不要修改 tests/test_calc.py，"
        "并保留其他现有功能。"
    )
    session.turn_required_language_suffix = None
    session.turn_language_change_from = None
    contract = StudioAgent._build_task_contract(session, message)
    keys = {item.key for item in contract.requirements}

    assert {
        "workspace_change",
        "target_file",
        "protected_path",
        "preserve_behavior",
    } <= keys
    assert "feature" not in keys
    assert any(
        item.key == "target_file" and item.expected == "calculator.py"
        for item in contract.requirements
    )
    assert any(
        item.key == "protected_path" and item.expected == "tests/test_calc.py"
        for item in contract.requirements
    )


def test_tool_failure_classification_prescribes_changed_strategy() -> None:
    category, strategy, retryable = StudioAgent._classify_tool_failure(
        "command", "python: can't open file 'missing.py': No such file", "python missing.py"
    )
    assert category == "missing_path"
    assert "搜索同名或近似文件" in strategy
    assert retryable is True

    category, strategy, retryable = StudioAgent._classify_tool_failure(
        "command", "HTTP 401 unauthorized", "client request"
    )
    assert category == "authentication"
    assert "停止重试" in strategy
    assert retryable is False


def test_budget_extends_only_after_meaningful_progress(tmp_path: Path) -> None:
    model = SequenceStudioModel(
        [
            *[
                StudioDecision(
                    action=StudioAction.SEARCH,
                    rationale=f"搜索线索 {index}",
                    query=f"missing-symbol-{index}",
                )
                for index in range(29)
            ],
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="汇总调查",
                message="调查完成。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "adaptive-budget.sqlite3")
    session = StudioSession(
        session_id="adaptive-budget",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    result = asyncio.run(StudioAgent(model, store, max_steps=40).handle(session, "调查并介绍项目"))

    assert result.status == "idle"
    assert result.turn_budget > 28
    assert any(
        event["event_type"] == "budget_extended" for event in store.events(result.session_id)
    )


def test_open_after_writing_is_complete_can_use_direct_launch(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("print('done')\n", encoding="utf-8")
    session = StudioSession(
        session_id="open-written",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py"],
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), StudioStore(tmp_path / "open-written.sqlite3")).handle(
            session, "写完了，帮我打开"
        )
    )

    assert result.pending_permission is not None
    assert result.pending_permission.command[-1] == "calculator.py"


def test_explicit_go_request_requires_a_go_file_before_finish(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "go-language-gate.sqlite3")
    session = StudioSession(
        session_id="go-language-gate",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        turn_changed_files=["calculator.py"],
        changed_files=["calculator.py"],
        turn_required_language_suffix=".go",
        verification_passed=True,
    )

    terminal = StudioAgent(ModelMustNotBeCalled(), store)._execute(
        session,
        SafeWorkspace(tmp_path),
        StudioDecision(action="finish", rationale="完成", message="计算器已完成。"),
    )

    assert terminal is False
    assert session.observations[-1].kind == "requirement_gate"
    assert ".go" in session.observations[-1].summary


def test_finish_is_blocked_when_requested_language_did_not_change(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("print('new')\n", encoding="utf-8")
    store = StudioStore(tmp_path / "language-gate.sqlite3")
    session = StudioSession(
        session_id="language-gate",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        turn_changed_files=["calculator.py"],
        changed_files=["calculator.py"],
        turn_language_change_from=".py",
        verification_passed=True,
    )
    workspace = SafeWorkspace(tmp_path)

    terminal = StudioAgent(ModelMustNotBeCalled(), store)._execute(
        session,
        workspace,
        StudioDecision(action="finish", rationale="完成", message="Python 版本已经完成。"),
    )

    assert terminal is False
    assert session.status != "completed"
    assert session.observations[-1].kind == "requirement_gate"
    assert "更换实现语言" in session.observations[-1].summary


def test_language_change_requirement_accepts_a_different_source_extension(tmp_path: Path) -> None:
    (tmp_path / "calculator.js").write_text("console.log('ok')\n", encoding="utf-8")
    store = StudioStore(tmp_path / "language-complete.sqlite3")
    session = StudioSession(
        session_id="language-complete",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        turn_changed_files=["calculator.js"],
        changed_files=["calculator.js"],
        turn_language_change_from=".py",
        verification_passed=True,
        observations=[
            StudioObservation(
                kind="create",
                summary="已创建 JavaScript 版本。",
                payload={
                    "path": "calculator.js",
                    "intent": "改用 JavaScript 实现",
                    "before": "",
                    "after": "console.log('ok')\n",
                    "diff": (
                        "--- a/calculator.js\n+++ b/calculator.js\n"
                        "@@ -0,0 +1 @@\n+console.log('ok')\n"
                    ),
                },
            )
        ],
    )
    workspace = SafeWorkspace(tmp_path)

    terminal = StudioAgent(ModelMustNotBeCalled(), store)._execute(
        session,
        workspace,
        StudioDecision(action="finish", rationale="完成", message="JavaScript 版本已完成。"),
    )

    assert terminal is True
    assert session.status == "completed"
    assert session.messages[-1].content.startswith("任务完成\n\n")
    assert "calculator.js" in session.messages[-1].content
    assert "JavaScript 版本已完成" in session.messages[-1].content


def test_regular_content_change_is_not_misclassified_as_language_change(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text('print("hello")\n', encoding="utf-8")
    session = StudioSession(
        session_id="content-change",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        changed_files=["hello.py"],
    )

    detected = StudioAgent._language_change_source_suffix(
        session,
        "把 hello.py 改成输出‘hello RAgent’，修改后重新读取并运行验证。",
        ["hello.py"],
    )

    assert detected is None


def test_opening_an_already_modified_app_remains_a_direct_launch(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("print('done')\n", encoding="utf-8")
    session = StudioSession(
        session_id="open-modified",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py"],
        context_estimated_tokens=1_234,
        context_actual_input_tokens=1_321,
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), StudioStore(tmp_path / "open-modified.sqlite3")).handle(
            session, "打开修改后的计算器"
        )
    )

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.command[-1] == "calculator.py"
    assert result.context_estimated_tokens == 1_234
    assert result.context_actual_input_tokens == 1_321


def test_reopen_prefers_most_recent_created_artifact(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "calculator.py").write_text("print('old')\n", encoding="utf-8")
    (tmp_path / "calculator.go").write_text("package main\n", encoding="utf-8")
    monkeypatch.setattr("veripatch.studio_agent.shutil.which", lambda name: "C:/Go/bin/go.exe")
    session = StudioSession(
        session_id="reopen-recent-go",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py", "calculator.go"],
        observations=[
            StudioObservation(
                kind="create",
                summary="Created the Go version.",
                payload={"path": "calculator.go"},
            )
        ],
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), StudioStore(tmp_path / "reopen-go.sqlite3")).handle(
            session, "再打开"
        )
    )

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.command == [
        "C:/Go/bin/go.exe",
        "run",
        "calculator.go",
    ]


def test_reopen_go_without_runtime_requests_install_and_launch_permission(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "calculator.py").write_text("print('old')\n", encoding="utf-8")
    (tmp_path / "calculator.go").write_text("package main\n", encoding="utf-8")
    monkeypatch.setattr("veripatch.studio_agent._find_go_executable", lambda: None)
    session = StudioSession(
        session_id="reopen-go-no-runtime",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py", "calculator.go"],
        observations=[
            StudioObservation(
                kind="create",
                summary="Created the Go version.",
                payload={"path": "calculator.go"},
            )
        ],
    )

    result = asyncio.run(
        StudioAgent(
            ModelMustNotBeCalled(), StudioStore(tmp_path / "reopen-go-blocked.sqlite3")
        ).handle(session, "再打开")
    )

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.command[:4] == ["winget", "install", "--id", "GoLang.Go"]
    assert result.pending_permission.capability == "install:go"
    assert result.pending_permission.follow_up_command == ["go", "run", "calculator.go"]
    assert "安装官方 Go" in result.pending_permission.reason
    assert "自动启动程序" in result.pending_permission.reason


def test_permission_resume_skips_direct_intent_routing(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    store = StudioStore(tmp_path / "permission-resume.sqlite3")
    session = StudioSession(
        session_id="permission-resume",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        plan=StudioAgent._build_plan(VerificationMode.AUTO),
    )
    model = SequenceStudioModel(
        [StudioDecision(action="respond", rationale="完成授权后的原任务", message="继续处理。")]
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "帮我打开",
            continuation=True,
            record_user_message=False,
            resume_after_permission=True,
        )
    )

    assert result.pending_permission is None
    assert result.messages[-1].content == "继续处理。"


def test_resume_with_verified_changes_continues_when_contract_is_still_unmet(
    tmp_path: Path,
) -> None:
    (tmp_path / "calculator.go").write_text("package main\n", encoding="utf-8")
    store = StudioStore(tmp_path / "resume-contract.sqlite3")
    session = StudioSession(
        session_id="resume-contract",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.go"],
        turn_changed_files=["calculator.go"],
        verification_passed=True,
        plan=StudioAgent._build_plan(VerificationMode.AUTO),
        task_contract=StudioTaskContract(
            objective="写 Go 计算器并打开",
            requirements=[
                StudioRequirement(key="launch_after_change", description="验证后打开新产物")
            ],
        ),
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action="respond",
                rationale="继续未完成的启动目标",
                message="正在继续处理启动步骤。",
            ),
            StudioDecision(
                action="respond",
                rationale="启动目标仍未完成",
                message="正在继续处理启动步骤。",
            ),
        ]
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "写 Go 计算器并打开",
            continuation=True,
            record_user_message=False,
            resume_after_permission=True,
        )
    )

    assert result.status == "paused"
    assert result.messages[-1].content != "正在继续处理启动步骤。"


def test_previously_approved_open_request_returns_result_to_model(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    command = ["cmd", "/c", "start", "", "index.html"]
    session = StudioSession(
        session_id="approved-launch",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["index.html"],
        approved_commands=[command],
    )
    runner = RunnerOutcome(
        command=command, exit_code=0, stdout="", stderr="", duration_seconds=0,
        launch_state="dispatched", window_confirmed=None,
    )

    model = SequenceStudioModel([
        StudioDecision(
            action="respond", rationale="只确认打开请求已发出",
            message="已交给系统打开 index.html，尚未确认浏览器窗口。",
        )
    ])
    with patch("veripatch.studio_tools.SafeStudioCommandRunner.launch", return_value=runner) as launch:
        result = asyncio.run(
            StudioAgent(model, StudioStore(tmp_path / "launch.sqlite3")).handle(
                session, "帮我打开"
            )
        )

    assert result.status == "idle"
    assert result.messages[-1].content == "已交给系统打开 index.html，尚未确认浏览器窗口。"
    assert not model.decisions
    launch.assert_called_once()
    assert not any(item.kind == "duplicate_action" for item in result.observations)


def test_previously_approved_gui_launch_without_window_is_not_reported_as_open(
    tmp_path: Path,
) -> None:
    (tmp_path / "calculator.py").write_text("print('no gui')\n", encoding="utf-8")
    command = ["pythonw.exe", "calculator.py"]
    session = StudioSession(
        session_id="approved-launch-without-window",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["calculator.py"],
        approved_commands=[command],
    )
    runner = RunnerOutcome(
        command=command,
        exit_code=1,
        stdout="Process 1234 did not create a visible window",
        stderr="未检测到目标程序的可见窗口。",
        duration_seconds=8,
    )

    model = SequenceStudioModel([
        StudioDecision(action="respond", rationale="根据启动结果回答",
                       message="calculator.py 未出现可见窗口，不能说已打开。")
    ])
    with patch("veripatch.studio_tools.SafeStudioCommandRunner.launch", return_value=runner):
        result = asyncio.run(
            StudioAgent(model, StudioStore(tmp_path / "no-window.sqlite3")).handle(
                session, "再打开一次"
            )
        )

    assert result.status == "idle"
    assert result.messages[-1].content == "calculator.py 未出现可见窗口，不能说已打开。"


def test_reopen_uses_recent_launch_target_instead_of_asking_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "calculator.py").write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr("veripatch.studio_agent.shutil.which", lambda name: "pythonw.exe")
    expected = ["pythonw.exe", "calculator.py"]
    session = StudioSession(
        session_id="reopen",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["index.html", "calculator.py"],
        observations=[
            StudioObservation(
                kind="command",
                summary="已启动",
                payload={
                    "command": [
                        "powershell",
                        "-Command",
                        "Start-Process -FilePath 'pythonw.exe' -ArgumentList 'calculator.py'",
                    ],
                    "exit_code": 0,
                },
            )
        ],
    )

    result = asyncio.run(
        StudioAgent(ModelMustNotBeCalled(), StudioStore(tmp_path / "reopen.sqlite3")).handle(
            session, "再打开一次"
        )
    )

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.command == expected


@pytest.mark.parametrize(
    "command",
    [
        ["cmd", "/c", "start", "", "python", "calculator.py"],
        ["powershell", "-NoProfile", "-Command", "Start-Process pythonw calculator.py"],
        ["C:\\Python313\\pythonw.exe", "calculator.py"],
    ],
)
def test_equivalent_windows_launch_commands_are_terminal(command: list[str]) -> None:
    assert is_detached_launch(command) is True


def test_verified_work_finishes_without_final_model_summary_call(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="verified-disconnect",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    model = VerifiedThenDisconnectedModel()
    result = asyncio.run(StudioAgent(model, store).handle(session, "创建并验证脚本"))

    assert result.status == "completed"
    assert result.verification_passed is True
    assert result.review_completed is True
    assert result.failure_reason is None
    assert model.calls == 2
    assert "本地验证已通过" in result.messages[-1].content
    assert not any(
        event["event_type"] == "recovered_completion"
        for event in store.events(session.session_id)
    )


def test_explicit_python_verification_continuation_skips_model(tmp_path: Path) -> None:
    (tmp_path / "probe.py").write_text('print("ok")\n', encoding="utf-8")
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="direct-verification",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        changed_files=["probe.py"],
        observations=[
            StudioObservation(
                kind="create",
                summary="已创建 probe.py。",
                payload={"diff": '--- a/probe.py\n+++ b/probe.py\n@@ -0,0 +1 @@\n+print("ok")\n'},
            )
        ],
    )
    model = SequenceStudioModel([])

    result = asyncio.run(
        StudioAgent(model, store).handle(session, "继续：运行 python probe.py 验证并完成。")
    )

    assert result.status == "completed"
    assert result.verification_passed is True
    assert result.usage.model_calls == 0
    assert "本地验证已通过" in result.messages[-1].content
    assert "python probe.py" not in result.messages[-1].content


def test_explicit_verification_runs_immediately_after_edit(tmp_path: Path) -> None:
    (tmp_path / "probe.py").write_text('print("old")\n', encoding="utf-8")
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="edit-direct-verification",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="更新输出",
                path="probe.py",
                old_text='print("old")',
                new_text='print("new")',
            )
        ]
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "把输出改为 new，然后运行 python probe.py 验证并完成。",
        )
    )

    assert result.status == "completed"
    assert result.verification_passed is True
    assert result.usage.model_calls == 1
    assert "本地验证已通过" in result.messages[-1].content
    assert "python probe.py" not in result.messages[-1].content


def test_static_web_creation_is_verified_and_completed_without_extra_model_call(
    tmp_path: Path,
) -> None:
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="static-web",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.CREATE,
                rationale="创建单文件网页",
                path="index.html",
                content="<!doctype html><html><body><h1>Calculator</h1></body></html>",
            )
        ]
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "创建网页计算器"))

    assert result.status == "completed"
    assert result.verification_passed is True
    assert result.usage.model_calls == 1
    assert any(item.kind == "static_web_check" for item in result.observations)


def test_static_web_creation_continues_when_launch_contract_remains(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="static-web-launch",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        task_contract=StudioTaskContract(
            objective="创建网页计算器并打开",
            requirements=[
                StudioRequirement(key="launch_after_change", description="验证后打开新产物")
            ],
        ),
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.CREATE,
                rationale="创建单文件网页",
                path="calculator.html",
                content="<!doctype html><html><body><h1>Calculator</h1></body></html>",
            ),
            StudioDecision(
                action=StudioAction.RUN_COMMAND,
                rationale="打开刚刚创建并验证的网页计算器",
                command=["cmd", "/c", "start", "", "calculator.html"],
            ),
        ]
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(session, "新建一个计算器，要有界面，然后打开")
    )

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.command[-1] == "calculator.html"
    assert result.usage.model_calls == 2
    assert any(item.kind == "requirement_gate" for item in result.observations)


def test_batch_actions_create_and_verify_in_one_model_call(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="batch-create-verify",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.BATCH,
                rationale="一次创建并验证",
                actions=[
                    StudioDecision(
                        action=StudioAction.CREATE,
                        rationale="创建脚本",
                        path="hello.py",
                        content='print("batch ok")\n',
                    ),
                    StudioDecision(
                        action=StudioAction.RUN_COMMAND,
                        rationale="验证脚本",
                        command=["python", "hello.py"],
                    ),
                ],
            )
        ]
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "创建并验证 hello.py"))

    assert result.status == "completed"
    assert result.verification_passed is True
    assert result.usage.model_calls == 1
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("batch ok")\n'


def test_non_allowlisted_command_requests_exact_user_permission(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="command-permission",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )
    command = ["python", "-c", "print('needs approval')"]
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.RUN_COMMAND,
                rationale="运行用户项目命令",
                command=command,
            )
        ]
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "运行项目命令"))

    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.access == "execute"
    assert result.pending_permission.command == command


def test_restores_before_snapshot_from_legacy_unified_diff() -> None:
    current = "first\nnew value\nlast\n"
    patch = """--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 first
-old value
+new value
 last
"""
    restored = _restore_before_from_unified_diff(current, patch, "example.py")
    assert restored == "first\nold value\nlast\n"
    assert _restore_before_from_unified_diff(current, patch, "other.py") is None


def test_studio_decision_normalizes_common_provider_format_drift() -> None:
    assert StudioDecision.normalize_common_model_variants("invalid") == "invalid"
    decision = StudioDecision.model_validate(
        {"action": "run_tests", "command": "python -m pytest -q"}
    )

    assert decision.rationale == "执行下一项经过审计的操作"
    assert decision.command == ["python", "-m", "pytest", "-q"]

    nullable_command = StudioDecision.model_validate(
        {
            "action": "read",
            "rationale": "读取实现",
            "path": "inventory_sync/service.py",
            "command": None,
        }
    )
    assert nullable_command.command == []

    memory_decision = StudioDecision(
        action=StudioAction.READ,
        rationale="验证缓存假设",
        path="inventory_sync/service.py",
        memory_update=StudioMemoryUpdate(
            hypotheses=["缓存键可能缺少仓库维度"],
            relevant_files=["inventory_sync/service.py"],
        ),
    )
    assert memory_decision.memory_update is not None
    assert memory_decision.memory_update.hypotheses == ["缓存键可能缺少仓库维度"]

    read = StudioDecision.model_validate(
        {
            "action": "read_file",
            "file_path": "inventory_sync/service.py",
            "reason": "检查实现",
        }
    )
    assert read.action is StudioAction.READ
    assert read.path == "inventory_sync/service.py"
    assert read.rationale == "检查实现"

    finished = StudioDecision.model_validate(
        {"action": "finish", "rationale": "测试已通过，修复完成。", "command": []}
    )
    assert finished.action is StudioAction.FINISH
    assert finished.message == "测试已通过，修复完成。"

    edit = StudioDecision.model_validate(
        {
            "action": "edit",
            "rationale": "扩展时间字段",
            "edits": [
                {
                    "file": "inventory_sync/parser.py",
                    "old": ["occurred_at=", "datetime.fromisoformat(timestamp),"],
                    "replacement": ["occurred_at=", "parse_timestamp(timestamp),"],
                }
            ],
        }
    )
    assert edit.action is StudioAction.EDIT
    assert edit.path == "inventory_sync/parser.py"
    assert edit.old_text == "occurred_at=\ndatetime.fromisoformat(timestamp),"
    assert edit.new_text == "occurred_at=\nparse_timestamp(timestamp),"

    created = StudioDecision.model_validate(
        {
            "action": "write_file",
            "file": "calculator.py",
            "reason": "创建 Python 图形程序",
            "code": "print('ready')\n",
        }
    )
    assert created.action is StudioAction.CREATE
    assert created.path == "calculator.py"
    assert created.content == "print('ready')\n"


def test_legacy_disabled_reasoning_is_migrated_to_low() -> None:
    session = StudioSession(
        session_id="legacy-reasoning",
        repo_root=".",
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="none",
    )
    assert session.reasoning_effort == "low"


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repository / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repository


def test_studio_agent_pauses_for_explicit_external_read_permission(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    external = tmp_path / "codex.log"
    external.write_text("diagnostic", encoding="utf-8")
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.REQUEST_PERMISSION,
                rationale="需要读取 Codex 日志定位启动失败",
                path=str(external),
            )
        ]
    )
    store = StudioStore(tmp_path / "permission.sqlite3")
    session = StudioSession(
        session_id="permission-run",
        repo_root=str(repository),
        provider="deepseek",
        model="deepseek-v4-flash",
        reasoning_effort="high",
    )
    result = asyncio.run(StudioAgent(model, store).handle(session, "帮我检查 Codex 为什么打不开"))
    assert result.status == "waiting_permission"
    assert result.pending_permission is not None
    assert result.pending_permission.path == str(external.resolve())
    assert result.approved_paths == []


def test_workspace_absolute_path_is_normalized_and_read_without_permission(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    target = repository / "calc.py"
    decision = StudioDecision(
        action=StudioAction.REQUEST_PERMISSION,
        rationale="读取计算器实现",
        path=str(target),
    )
    workspace = SafeWorkspace(repository)
    StudioAgent._normalize_workspace_decision_path(workspace, decision)
    session = StudioSession(
        session_id="internal-absolute-path",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    terminal = StudioAgent(
        ModelMustNotBeCalled(), StudioStore(tmp_path / "paths.sqlite3")
    )._execute(session, workspace, decision)

    assert decision.path == "calc.py"
    assert terminal is False
    assert session.pending_permission is None
    assert session.observations[-1].kind == "read"
    assert session.observations[-1].payload["path"] == "calc.py"
    assert session.observations[-1].payload["permission_required"] is False


def test_read_action_supports_workspace_absolute_and_relative_paths(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    workspace = SafeWorkspace(repository)
    agent = StudioAgent(ModelMustNotBeCalled(), StudioStore(tmp_path / "read-paths.sqlite3"))
    session = StudioSession(
        session_id="read-paths",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    for supplied in ("calc.py", str(repository / "calc.py")):
        decision = StudioDecision(
            action=StudioAction.READ,
            rationale="读取实现",
            path=supplied,
        )
        agent._normalize_workspace_decision_path(workspace, decision)
        assert agent._execute(session, workspace, decision) is False
        assert decision.path == "calc.py"
        assert session.observations[-1].payload["path"] == "calc.py"


def test_create_for_existing_file_is_normalized_to_atomic_edit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    workspace = SafeWorkspace(repository)
    decision = StudioDecision(
        action=StudioAction.CREATE,
        rationale="重新制作这个文件",
        path="calc.py",
        content="def add(a, b):\n    return a + b\n",
    )

    StudioAgent._normalize_file_action(workspace, decision)

    assert decision.action is StudioAction.EDIT
    assert decision.old_text == "def add(a, b):\n    return a - b\n"
    assert decision.new_text == "def add(a, b):\n    return a + b\n"
    assert decision.content is None


def test_create_for_empty_existing_file_uses_atomic_edit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "empty.py").write_text("", encoding="utf-8")
    workspace = SafeWorkspace(repository)
    decision = StudioDecision(
        action=StudioAction.CREATE,
        rationale="填写已有空文件",
        path="empty.py",
        content="value = 1\n",
    )

    StudioAgent._normalize_file_action(workspace, decision)

    assert decision.action is StudioAction.EDIT
    assert decision.old_text == ""
    assert workspace.apply_edits(
        [FileEdit(path=decision.path, old_text=decision.old_text, new_text=decision.new_text)],
        protect_tests=False,
    ) == ["empty.py"]
    assert (repository / "empty.py").read_text(encoding="utf-8") == "value = 1\n"


def test_task_state_progress_only_contains_current_turn_observations(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="turn-local-state",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        observations=[
            StudioObservation(kind="delete", summary="上轮已删除 index.html。"),
            StudioObservation(kind="edit", summary="本轮已修改 index.html。"),
        ],
        turn_observation_start=1,
    )
    contract = StudioAgent._build_task_contract(session, "重新弄个新网页")
    session.task_state = StudioAgent._build_task_state(session, contract, "重新弄个新网页")

    StudioAgent._refresh_task_state(session)

    assert session.task_state.completed_actions == ["本轮已修改 index.html。"]


def test_studio_api_can_deny_a_pending_permission(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "permission-api.sqlite3")
    client = TestClient(create_app(settings))
    created = client.post(
        "/studio-api/sessions",
        json={
            "repo_root": str(repository),
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
        },
    ).json()
    store = StudioStore(settings.database_path)
    session = store.load(created["session_id"])
    assert session is not None
    session.status = "waiting_permission"
    session.pending_permission = StudioPermissionRequest(
        request_id="permission-id",
        path=str(tmp_path / "outside.log"),
        reason="读取诊断日志",
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())
    response = client.post(
        f"/studio-api/sessions/{session.session_id}/permissions/permission-id",
        json={"approved": False},
    )
    assert response.json() == {"status": "denied"}
    denied = store.load(session.session_id)
    assert denied is not None
    assert denied.pending_permission is None
    assert denied.approved_paths == []


def test_write_permission_request_under_read_approved_parent_still_prompts(
    tmp_path: Path,
) -> None:
    store = StudioStore(tmp_path / "covered-permission.sqlite3")
    approved = tmp_path / "approved"
    approved.mkdir()
    session = StudioSession(
        session_id="covered-permission",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        approved_paths=[str(approved)],
    )

    terminal = StudioAgent(ModelMustNotBeCalled(), store)._execute(
        session,
        SafeWorkspace(tmp_path, approved_roots=[approved]),
        StudioDecision(
            action=StudioAction.REQUEST_PERMISSION,
            rationale="创建安装目录",
            path=str(approved / "Calculator"),
            access="write",
        ),
    )

    assert terminal is True
    assert session.pending_permission is not None
    assert session.pending_permission.access == "write"
    assert session.pending_permission.operation is None


def test_permission_request_for_creation_is_labeled_write(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "write-permission.sqlite3")
    session = StudioSession(
        session_id="write-permission",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
    )

    terminal = StudioAgent(ModelMustNotBeCalled(), store)._execute(
        session,
        SafeWorkspace(tmp_path),
        StudioDecision(
            action=StudioAction.REQUEST_PERMISSION,
            rationale="需要创建安装目录并写入文件",
            path=str(tmp_path.parent / "outside-new"),
            access="write",
        ),
    )

    assert terminal is True
    assert session.pending_permission is not None
    assert session.pending_permission.access == "write"


def test_studio_api_can_revise_a_pending_permission_with_user_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "permission-revision.sqlite3")
    store = StudioStore(settings.database_path)
    session = StudioSession(
        session_id="permission-revision",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        pending_permission=StudioPermissionRequest(
            request_id="revision-id",
            path=str(repository),
            reason="install Go",
            access="execute",
            command=["winget", "install", "--id", "GoLang.Go"],
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())

    async def revised_handle(_self, current, content, **_kwargs):
        assert content == "调整执行方案：安装到 D 盘"
        current.status = "idle"
        store.save(current, "assistant_message", {"content": "已收到调整要求。"})
        return current

    monkeypatch.setattr("veripatch.studio_api.StudioAgent.handle", revised_handle)
    client = TestClient(create_app(settings))

    response = client.post(
        "/studio-api/sessions/permission-revision/permissions/revision-id",
        json={"approved": False, "instruction": "安装到 D 盘"},
    )

    assert response.json() == {"status": "revising"}
    revised = store.load("permission-revision")
    assert revised is not None
    assert revised.pending_permission is None
    assert revised.messages == []
    assert any(
        event["event_type"] == "permission_revised" for event in store.events("permission-revision")
    )


def test_permission_revision_does_not_duplicate_recent_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "permission-revision-duplicate.sqlite3")
    store = StudioStore(settings.database_path)
    content = "调整执行方案：安装到 D 盘"
    session = StudioSession(
        session_id="permission-revision-duplicate",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        messages=[StudioMessage(role="user", content=content)],
        pending_permission=StudioPermissionRequest(
            request_id="revision-id",
            path=str(repository),
            reason="install Go",
            access="execute",
            command=["winget", "install", "--id", "GoLang.Go"],
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())

    async def revised_handle(_self, current, _content, **_kwargs):
        current.status = "idle"
        store.save(current, "assistant_message", {"content": "已调整。"})
        return current

    monkeypatch.setattr("veripatch.studio_api.StudioAgent.handle", revised_handle)
    client = TestClient(create_app(settings))
    response = client.post(
        "/studio-api/sessions/permission-revision-duplicate/permissions/revision-id",
        json={"approved": False, "instruction": "安装到 D 盘"},
    )

    assert response.json() == {"status": "revising"}
    revised = store.load("permission-revision-duplicate")
    assert revised is not None
    assert [message.content for message in revised.messages] == [content]


def test_studio_api_recovers_a_running_session_after_process_restart(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "restart.sqlite3")
    store = StudioStore(settings.database_path)
    session = StudioSession(
        session_id="interrupted",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="running",
    )
    store.save(session, "user_message", {"content": "work"})

    TestClient(create_app(settings))

    recovered = store.load(session.session_id)
    assert recovered is not None
    assert recovered.status == "idle"
    assert store.events(session.session_id)[-1]["event_type"] == "interrupted"


def test_studio_api_recovers_a_consumed_permission_after_process_restart(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "permission-restart.sqlite3")
    store = StudioStore(settings.database_path)
    session = StudioSession(
        session_id="stale-permission",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        pending_permission=None,
    )
    store.save(session, "permission_approved", {"command": ["python", "-V"]})

    TestClient(create_app(settings))

    recovered = store.load(session.session_id)
    assert recovered is not None
    assert recovered.status == "idle"
    assert recovered.activity == "idle"
    assert store.events(session.session_id)[-1]["event_type"] == "permission_recovered"


def test_studio_api_removes_legacy_internal_permission_messages_from_chat(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "legacy-message.sqlite3")
    store = StudioStore(settings.database_path)
    session = StudioSession(
        session_id="legacy-message",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        messages=[
            StudioMessage(role="user", content="改进界面"),
            StudioMessage(role="user", content="用户已批准并已执行命令 python -c pass。结果：0"),
        ],
    )
    store.save(session, "created", {})

    response = TestClient(create_app(settings)).get("/studio-api/sessions/legacy-message")

    assert response.status_code == 200
    assert [item["content"] for item in response.json()["messages"]] == ["改进界面"]
    persisted = store.load("legacy-message")
    assert persisted is not None
    assert [item.content for item in persisted.messages] == ["改进界面"]


def test_studio_api_resumes_after_approved_command_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "permission-failure.sqlite3")
    client = TestClient(create_app(settings))
    store = StudioStore(settings.database_path)
    command = ["python", "-c", "value = 1; print(value)"]
    session = StudioSession(
        session_id="permission-failure",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        pending_permission=StudioPermissionRequest(
            request_id="failure-id",
            path=str(repository),
            reason="run",
            access="execute",
            command=command,
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())

    def fail_to_start(_self: object, _command: list[str]) -> RunnerOutcome:
        raise OSError("process could not start")

    monkeypatch.setattr("veripatch.studio_api.SafeStudioCommandRunner.run", fail_to_start)

    response = client.post(
        "/studio-api/sessions/permission-failure/permissions/failure-id",
        json={"approved": True},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "recovering"}
    recovered = store.load("permission-failure")
    assert recovered is not None
    assert recovered.pending_permission is None
    assert command not in recovered.approved_commands  # 本次允许不能变成会话级授权
    assert not any("process could not start" in item.content for item in recovered.messages)
    events = store.events("permission-failure")
    failure = next(item for item in events if item["event_type"] == "permission_execution_failed")
    assert failure["payload"]["resuming"] is True
    observation = next(
        item
        for item in events
        if item["event_type"] == "observation" and item["payload"].get("kind") == "tool_error"
    )
    assert observation["payload"]["payload"]["error_type"] == "OSError"


def test_studio_api_executes_an_approved_detached_launch_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "launch.sqlite3")
    client = TestClient(create_app(settings))
    store = StudioStore(settings.database_path)
    command = ["cmd", "/c", "start", "", "python", "calculator_gui.py"]
    session = StudioSession(
        session_id="launch",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        pending_permission=StudioPermissionRequest(
            request_id="launch-id",
            path=str(repository),
            reason="launch",
            access="execute",
            command=command,
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())
    launch_calls = 0
    model = CapturingStudioModel([
        StudioDecision(action="run_command", rationale="尝试再次启动",
                       command=command),
        StudioDecision(action="respond", rationale="根据启动证据回答",
                       message="已确认程序窗口打开。")
    ])
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *_: model)

    def launch_once(_self: object, value: list[str]) -> RunnerOutcome:
        nonlocal launch_calls
        launch_calls += 1
        return RunnerOutcome(
            command=value, exit_code=0, stdout="", stderr="", duration_seconds=0,
            launch_state="window_confirmed", window_confirmed=True,
        )

    monkeypatch.setattr("veripatch.studio_api.SafeStudioCommandRunner.launch", launch_once)

    response = client.post(
        "/studio-api/sessions/launch/permissions/launch-id",
        json={"approved": True, "scope": "session"},
    )
    duplicate = client.post(
        "/studio-api/sessions/launch/permissions/launch-id", json={"approved": True}
    )

    assert response.json() == {"status": "approved"}
    assert duplicate.status_code == 409
    assert launch_calls == 1
    completed = store.load("launch")
    assert completed is not None
    assert completed.status == "idle"
    assert completed.pending_permission is None
    assert completed.approved_capabilities == ["launch:calculator_gui.py"]
    assert completed.messages[-1].content == "已确认程序窗口打开。"
    assert not model.decisions
    assert model.contexts[0]["latest_tool_result"]["payload"]["launch_state"] == "window_confirmed"
    assert model.contexts[0]["latest_tool_result"]["payload"]["window_confirmed"] is True


@pytest.mark.parametrize(
    ("launch_state", "window_confirmed", "stdout"),
    [
        ("window_confirmed", True, "Started process 123; window_confirmed=true"),
        ("dispatched", None, "Windows accepted open request for calculator.html"),
    ],
)
def test_studio_api_completes_compound_task_after_approved_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    launch_state: str, window_confirmed: bool | None, stdout: str,
) -> None:
    repository = _repository(tmp_path)
    (repository / "calculator.html").write_text("<html></html>", encoding="utf-8")
    settings = Settings(database_path=tmp_path / "compound-launch.sqlite3")
    client = TestClient(create_app(settings))
    store = StudioStore(settings.database_path)
    command = ["cmd", "/c", "start", "", "calculator.html"]
    session = StudioSession(
        session_id="compound-launch",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        turn_changed_files=["calculator.html"],
        changed_files=["calculator.html"],
        verification_passed=True,
        task_contract=StudioTaskContract(
            objective="创建计算器，验证后打开",
            requirements=[
                StudioRequirement(key="workspace_change", description="创建计算器"),
                StudioRequirement(key="verification", description="验证计算器"),
                StudioRequirement(key="launch_after_change", description="验证后打开新产物"),
            ],
        ),
        pending_permission=StudioPermissionRequest(
            request_id="compound-launch-id",
            path=str(repository),
            reason="open calculator",
            access="execute",
            command=command,
        ),
    )
    session.observations.append(
        StudioObservation(
            kind="create",
            summary="已创建 calculator.html。",
            payload={
                "path": "calculator.html", "intent": "创建图形计算器",
                "diff": "--- a/calculator.html\n+++ b/calculator.html\n+<html></html>\n",
            },
        )
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())
    model = SequenceStudioModel([
        StudioDecision(
            action="finish", rationale="根据已记录的启动结果完成任务",
            message=("已创建并验证 calculator.html，窗口已确认打开。"
                     if window_confirmed else
                     "已创建并验证 calculator.html，系统已接收打开请求，窗口尚未确认。"),
        )
    ])
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *_: model)
    monkeypatch.setattr(
        "veripatch.studio_api.SafeStudioCommandRunner.launch",
        lambda _self, value: RunnerOutcome(
            command=value,
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_seconds=0,
            launch_state=launch_state,
            window_confirmed=window_confirmed,
        ),
    )

    response = client.post(
        "/studio-api/sessions/compound-launch/permissions/compound-launch-id",
        json={"approved": True},
    )

    assert response.json() == {"status": "approved"}
    completed = store.load("compound-launch")
    assert completed is not None
    assert completed.status == "completed"
    assert completed.activity == "completed"
    assert completed.pending_permission is None
    assert completed.messages[-1].content.count("calculator.html") >= 1
    assert not StudioAgent._validate_task_contract(completed)
    assert not model.decisions
    if launch_state == "dispatched":
        assert "窗口尚未确认" in completed.messages[-1].content
        assert completed.observations[-1].payload.get("launch_state") == "dispatched" or any(
            item.payload.get("launch_state") == "dispatched" for item in completed.observations
        )


def test_studio_api_installs_missing_go_then_launches_requested_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    (repository / "calculator.go").write_text("package main\n", encoding="utf-8")
    settings = Settings(database_path=tmp_path / "install-go.sqlite3")
    client = TestClient(create_app(settings))
    store = StudioStore(settings.database_path)
    install_command = [
        "winget",
        "install",
        "--id",
        "GoLang.Go",
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--silent",
    ]
    session = StudioSession(
        session_id="install-go",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        pending_permission=StudioPermissionRequest(
            request_id="install-go-id",
            path=str(repository),
            reason="install Go and launch calculator.go",
            access="execute",
            command=install_command,
            follow_up_command=["go", "run", "calculator.go"],
            capability="install:go",
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())
    monkeypatch.setattr(
        "veripatch.studio_api.SafeStudioCommandRunner.run",
        lambda _self, value: RunnerOutcome(
            command=value, exit_code=0, stdout="installed", stderr="", duration_seconds=1
        ),
    )
    launched: list[list[str]] = []

    def launch(_self, value):
        launched.append(value)
        return RunnerOutcome(command=value, exit_code=0, stdout="", stderr="", duration_seconds=0)

    monkeypatch.setattr("veripatch.studio_api.SafeStudioCommandRunner.launch", launch)
    monkeypatch.setattr("veripatch.studio_api._go_executable", lambda: "C:/Go/bin/go.exe")

    response = client.post(
        "/studio-api/sessions/install-go/permissions/install-go-id",
        json={"approved": True},
    )

    assert response.json() == {"status": "approved"}
    assert launched == [["C:/Go/bin/go.exe", "run", "calculator.go"]]
    completed = store.load("install-go")
    assert completed is not None
    assert completed.status == "idle"
    assert completed.messages[-1].content == "Go 已安装，并已启动 calculator.go。"


def test_unverified_launch_stops_without_automatic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "launch-not-running.sqlite3")
    client = TestClient(create_app(settings))
    store = StudioStore(settings.database_path)
    command = ["cmd", "/c", "start", "", "python", "calculator_gui.py"]
    session = StudioSession(
        session_id="launch-not-running",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        pending_permission=StudioPermissionRequest(
            request_id="launch-id",
            path=str(repository),
            reason="launch",
            access="execute",
            command=command,
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())
    model = SequenceStudioModel([
        StudioDecision(action="respond", rationale="如实说明启动证据",
                       message="启动命令结束，但尚未确认目标程序仍在运行。")
    ])
    monkeypatch.setattr("veripatch.studio_api.StudioProviderModel", lambda *_: model)
    monkeypatch.setattr(
        "veripatch.studio_api.SafeStudioCommandRunner.launch",
        lambda _self, value: RunnerOutcome(
            command=value,
            exit_code=0,
            stdout="Started process 123",
            stderr="",
            duration_seconds=0,
        ),
    )

    response = client.post(
        "/studio-api/sessions/launch-not-running/permissions/launch-id",
        json={"approved": True},
    )

    assert response.json() == {"status": "approved"}
    completed = store.load("launch-not-running")
    assert completed is not None
    assert completed.status == "idle"
    assert completed.messages[-1].content == "启动命令结束，但尚未确认目标程序仍在运行。"
    assert not model.decisions


def test_studio_api_returns_live_process_output_without_reusing_old_code_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    settings = Settings(database_path=tmp_path / "processes.sqlite3")
    client = TestClient(create_app(settings))
    store = StudioStore(settings.database_path)
    command = ["cmd", "/c", "tasklist"]
    session = StudioSession(
        session_id="processes",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="low",
        status="waiting_permission",
        changed_files=["calculator_gui.py"],
        verification_passed=True,
        pending_permission=StudioPermissionRequest(
            request_id="process-id",
            path=str(repository),
            reason="inspect",
            access="execute",
            command=command,
        ),
    )
    store.save(session, "permission_requested", session.pending_permission.model_dump())
    monkeypatch.setattr(
        "veripatch.studio_api.SafeStudioCommandRunner.run",
        lambda _self, value: RunnerOutcome(
            command=value,
            exit_code=0,
            stdout="RAgent.exe  100\npython.exe  200\n",
            stderr="",
            duration_seconds=0,
        ),
    )

    response = client.post(
        "/studio-api/sessions/processes/permissions/process-id", json={"approved": True}
    )

    assert response.json() == {"status": "approved"}
    inspected = store.load("processes")
    assert inspected is not None
    assert inspected.status == "idle"
    assert "RAgent.exe" in inspected.messages[-1].content
    assert "已修改" not in inspected.messages[-1].content
    assert "验证通过" not in inspected.messages[-1].content


def test_studio_agent_runs_read_edit_test_finish_loop(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="先阅读", path="calc.py"),
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="修复运算符",
                path="calc.py",
                old_text="return a - b",
                new_text="return a + b",
            ),
            StudioDecision(
                action=StudioAction.RUN_TESTS,
                rationale="验证修改",
                command=["python", "-m", "pytest", "-q"],
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="测试通过",
                message="已经修复并通过测试。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "studio.sqlite3")
    session = StudioSession(
        session_id="studio-1",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
    )
    result = asyncio.run(StudioAgent(model, store).handle(session, "修复加法函数"))
    assert result.status == "completed"
    assert result.step == 3
    assert result.usage.model_calls == 3
    assert result.changed_files == ["calc.py"]
    assert result.observations[-3].kind == "test"
    assert result.observations[-3].payload["exit_code"] == 0
    assert result.observations[-2].kind == "result_review"
    assert result.observations[-1].kind == "final_review"
    assert result.review_completed is True
    assert all(item.status == "completed" for item in result.plan)
    assert 16 <= result.turn_budget <= 60
    assert "return a + b" in (repository / "calc.py").read_text(encoding="utf-8")
    assert store.load("studio-1") is not None
    events = store.events("studio-1")
    assert len(events) >= 8
    edit_payload = next(
        event["payload"]["payload"]
        for event in events
        if event["event_type"] == "observation" and event["payload"]["kind"] == "edit"
    )
    assert "return a - b" in edit_payload["before"]
    assert "return a + b" in edit_payload["after"]
    assert "-    return a - b" in edit_payload["diff"]


def test_strict_studio_requires_baseline_change_and_fixed_verification(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="尝试提前完成",
                message="完成",
            ),
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="修复实现",
                path="calc.py",
                old_text="return a - b",
                new_text="return a + b",
            ),
            StudioDecision(
                action=StudioAction.RUN_TESTS,
                rationale="验证修复",
                command=["python", "-m", "pytest", "test_calc.py", "-q"],
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="证据齐全",
                message="严格验证完成。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "strict.sqlite3")
    session = StudioSession(
        session_id="strict-1",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        verification_mode=VerificationMode.STRICT,
        test_command=["python", "-m", "pytest", "-q"],
    )
    result = asyncio.run(StudioAgent(model, store).handle(session, "严格修复加法"))
    assert result.status == "completed"
    assert result.baseline_reproduced is True
    assert result.verification_passed is True
    assert result.changed_files == ["calc.py"]
    assert [item.kind for item in result.observations] == [
        "baseline_test",
        "verification_gate",
        "edit",
        "test",
        "result_review",
        "final_review",
    ]
    verified_command = result.observations[-3].payload["command"]
    assert verified_command[1:] == ["-m", "pytest", "-q"]
    assert Path(verified_command[0]).name in {"python", "python.exe"}


def test_studio_agent_creates_file_and_runs_safe_command(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.CREATE,
                rationale="创建配置",
                path="src/settings.json",
                content='{"enabled": true}\n',
            ),
            StudioDecision(action=StudioAction.LIST_FILES, rationale="刷新文件列表"),
            StudioDecision(
                action=StudioAction.SEARCH,
                rationale="确认新配置内容",
                query="enabled",
            ),
            StudioDecision(
                action=StudioAction.RUN_COMMAND,
                rationale="检查工作区",
                command=["git", "status", "--short"],
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="完成",
                message="文件已创建，命令已执行。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "create.sqlite3")
    session = StudioSession(
        session_id="studio-create",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        verification_mode=VerificationMode.QUICK,
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "创建配置并检查状态"))

    assert result.status == "completed"
    assert result.changed_files == ["src/settings.json"]
    assert (repository / "src/settings.json").read_text(encoding="utf-8") == ('{"enabled": true}\n')
    assert [item.kind for item in result.observations] == [
        "create",
        "files",
        "search",
        "command",
        "result_review",
        "final_review",
    ]


def test_studio_agent_replans_context_after_failed_verification(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = RecordingStudioModel(
        [
            StudioDecision(
                action=StudioAction.RUN_TESTS,
                rationale="先复现失败",
                command=["python", "-m", "pytest", "-q"],
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="说明当前发现",
                message="已经复现失败，下一步需要定位实现。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "reflection.sqlite3")
    session = StudioSession(
        session_id="reflection-1",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "复现并分析失败"))

    assert result.status == "idle"
    assert result.observations[-1].kind == "test"
    assert result.observations[-1].payload["exit_code"] != 0
    assert "改变策略" in str(model.contexts[1]["instruction"])
    assert model.contexts[0]["agent_identity"] == {
        "name": "RAgent",
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
    }
    assert any(item.key == "verify" and item.status == "blocked" for item in result.plan)


def test_observation_assessment_replans_missing_path_to_different_action(
    tmp_path: Path,
) -> None:
    (tmp_path / "actual.py").write_text("value = 1\n", encoding="utf-8")
    model = RecordingStudioModel(
        [
            StudioDecision(
                action=StudioAction.READ,
                rationale="尝试读取用户提到的路径",
                path="missing.py",
            ),
            StudioDecision(
                action=StudioAction.LIST_FILES,
                rationale="根据失败证据查找真实路径",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="报告重规划结果",
                message="missing.py 不存在，已改为检查项目文件列表。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "observation-replan.sqlite3")
    session = StudioSession(
        session_id="observation-replan",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "分析 missing.py 的作用"))

    assessments = [
        event["payload"]
        for event in store.events(session.session_id)
        if event["event_type"] == "observation_assessed"
    ]
    assert assessments[0]["outcome"] == "failed_retryable"
    assert assessments[0]["category"] == "missing_path"
    assert assessments[0]["should_replan"] is True
    assert assessments[1]["outcome"] == "succeeded"
    assert result.task_state is not None
    assert result.task_state.replan_count == 1
    assert "确认真实路径" in str(model.contexts[1]["instruction"])
    assert "继续执行当前重规划策略" in str(model.contexts[2]["instruction"])
    assert result.task_state.current_strategy is not None
    assert any(item.kind == "files" for item in result.observations)


def test_failed_test_assessment_keeps_task_in_investigation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = RecordingStudioModel(
        [
            StudioDecision(
                action=StudioAction.RUN_TESTS,
                rationale="复现失败",
                command=["python", "-m", "pytest", "-q"],
            ),
            StudioDecision(
                action=StudioAction.READ,
                rationale="测试失败后读取相关实现",
                path="calc.py",
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="总结调查证据",
                message="已根据失败转入实现调查。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "test-failure-assessment.sqlite3")
    session = StudioSession(
        session_id="test-failure-assessment",
        repo_root=str(repository),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "复现并分析失败"))

    failed = next(
        event["payload"]
        for event in store.events(session.session_id)
        if event["event_type"] == "observation_assessed"
        and event["payload"]["outcome"] == "failed_retryable"
    )
    assert failed["category"] == "test_failure"
    assert failed["next_phase"] == "investigate"
    assert result.task_state is not None
    assert result.task_state.replan_count == 1
    assert any(item.kind == "read" for item in result.observations)


def test_pytest_cache_warning_does_not_hide_assertion_failure() -> None:
    output = (
        "FAILED test_calc.py::test_add - assert -1 == 5\n"
        "============================== warnings summary ==============================\n"
        "PytestCacheWarning: Permission denied: .pytest_cache\n"
    )
    category, _, retryable = StudioAgent._classify_tool_failure(
        "test", output, "python -m pytest -q"
    )
    assert category == "test_failure"
    assert retryable


def test_final_review_blocks_hardcoded_secret(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="加入配置",
                path="calc.py",
                old_text="def add(a, b):",
                new_text='API_KEY = "abcdefgh123"\n\ndef add(a, b):',
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="尝试完成",
                message="修改完成。",
            ),
            StudioDecision(
                action=StudioAction.FAIL,
                rationale="自审发现风险",
                message="最终自审发现硬编码凭据，需要先移除。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "review.sqlite3")
    session = StudioSession(
        session_id="review-1",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        verification_mode=VerificationMode.QUICK,
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "修改配置"))

    review = next(item for item in result.observations if item.kind == "final_review")
    assert review.payload["passed"] is False
    assert "硬编码凭据" in review.summary
    assert result.review_completed is False
    assert result.status == "failed"


def test_independent_result_review_corrects_unsupported_verification_claim(
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    model = SequenceStudioModel(
        [
            StudioDecision(action=StudioAction.READ, rationale="读取目标", path="hello.py"),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="总结",
                message="已分析 hello.py，测试已经通过。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "result-review.sqlite3")
    session = StudioSession(
        session_id="result-review",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "分析 hello.py"))

    assert result.status == "completed"
    review = next(item for item in result.observations if item.kind == "result_review")
    assert review.payload["verdict"] == "corrected"
    assert review.payload["verification_claim_grounded"] is False
    assert "测试已经通过" not in result.messages[-1].content
    assert "没有可核验的验证通过证据" in result.messages[-1].content


def test_independent_result_review_blocks_observed_denied_action(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="policy-review",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        task_state=StudioTaskState(
            objective="只分析，不运行命令",
            intent="analysis",
            denied_actions=[StudioAction.RUN_COMMAND.value],
        ),
        observations=[
            StudioObservation(
                kind="command",
                summary="命令执行完成。",
                payload={"command": ["python", "hello.py"], "exit_code": 0},
            )
        ],
    )

    review = StudioAgent._review_final_result(session, "分析完成。")

    assert review.verdict.value == "blocked"
    assert review.policy_compliant is False
    assert review.blockers == ["实际执行了禁止动作 run_command"]


def test_result_review_uses_fresh_cross_turn_verification_evidence(tmp_path: Path) -> None:
    session = StudioSession(
        session_id="cross-turn-verification",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
        verification_passed=False,
        observations=[
            StudioObservation(
                kind="test",
                summary="测试通过。",
                payload={
                    "command": ["python", "-m", "pytest", "-q"],
                    "exit_code": 0,
                    "stdout": "2 passed in 0.01s",
                },
            )
        ],
    )

    review = StudioAgent._review_final_result(session, "测试已经通过：2 passed in 0.01s。")

    assert review.verdict.value == "passed"
    assert review.verification_claim_grounded is True


def test_bare_number_after_completed_task_requests_clarification(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "bare-number.sqlite3")
    session = StudioSession(
        session_id="bare-number",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="deepseek-v4-pro",
        reasoning_effort="low",
        status="completed",
    )

    result = asyncio.run(StudioAgent(NeverDecideModel(), store).handle(session, "1"))

    assert result.activity == "waiting_user"
    assert "当前没有待选择的编号选项" in result.messages[-1].content
    assert any(
        event["event_type"] == "input_clarification" for event in store.events(session.session_id)
    )


def test_model_authentication_errors_do_not_expose_key_fragments() -> None:
    error = RuntimeError(
        "AuthenticationError: Error code: 401 - Incorrect API key provided: sk-secret-value"
    )

    message = StudioAgent._model_error_message(error)

    assert message == ("模型认证失败（HTTP 401）。请检查所选供应商、Base URL 和 API Key 是否匹配。")
    assert "sk-secret" not in message


def test_structured_action_error_is_not_reported_as_network_failure() -> None:
    error = RuntimeError("模型连续三次未返回有效的结构化动作，请重试本轮。")

    message = StudioAgent._model_error_message(error)

    assert "未遵循 Agent 动作协议" in message
    assert "检查网络" not in message


def test_http_status_error_reports_actual_gateway_status() -> None:
    import httpx

    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={"error": {"message": "reasoning_effort is not supported"}},
    )
    error = httpx.HTTPStatusError("Bad request", request=request, response=response)

    message = StudioAgent._model_error_message(error)

    assert "HTTP 400" in message
    assert "reasoning_effort is not supported" in message
    assert "检查网络" not in message


def test_http_status_error_hides_api_key_in_gateway_detail() -> None:
    import httpx

    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    response = httpx.Response(
        502,
        request=request,
        json={"error": {"message": "upstream rejected sk-super-secret-value"}},
    )
    error = httpx.HTTPStatusError("Bad gateway", request=request, response=response)

    message = StudioAgent._model_error_message(error)

    assert "HTTP 502" in message
    assert "sk-super-secret-value" not in message
    assert "[密钥已隐藏]" in message


def test_dynamic_budget_scales_with_task_complexity(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "budget.sqlite3")
    agent = StudioAgent(SequenceStudioModel([]), store, max_steps=20)

    simple = agent._dynamic_budget("解释这个函数", 8)
    complex_task = agent._dynamic_budget(
        "1. 修复并发问题\n2. 跨文件重构\n3. 增加恢复和安全测试\n4. 运行完整测试",
        200,
    )

    assert 8 <= simple < complex_task <= 20


def test_structured_memory_and_symbol_retrieval_persist_between_steps(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = RecordingStudioModel(
        [
            StudioDecision(
                action=StudioAction.READ,
                rationale="阅读 add 实现",
                path="calc.py",
                memory_update=StudioMemoryUpdate(
                    hypotheses=["add 的运算符可能错误"],
                    relevant_files=["calc.py"],
                ),
            ),
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="汇报定位结果",
                message="已经定位到 add 实现。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "memory.sqlite3")
    session = StudioSession(
        session_id="memory-1",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
    )

    result = asyncio.run(
        StudioAgent(model, store).handle(
            session,
            "请检查 add 函数。\n要求：必须保留函数签名。",
        )
    )

    first_retrieval = model.contexts[0]["retrieved_context"]
    assert isinstance(first_retrieval, dict)
    assert any(item["name"] == "add" for item in first_retrieval["symbols"])
    assert first_retrieval["snippets"][0]["path"] == "calc.py"
    assert first_retrieval["impact"]["tests"] == ["test_calc.py"]
    assert first_retrieval["relationships"]["imports"]
    assert first_retrieval["relationships"]["calls"]
    second_memory = model.contexts[1]["structured_memory"]
    assert isinstance(second_memory, dict)
    assert "要求：必须保留函数签名。" in result.memory.constraints
    assert "add 的运算符可能错误" in second_memory["hypotheses"]
    assert "calc.py" in result.memory.relevant_files
    assert any("已阅读 calc.py" in fact for fact in result.memory.facts)
    assert result.context_estimated_tokens > 0


def test_context_budget_trims_low_priority_payloads(tmp_path: Path) -> None:
    agent = StudioAgent(
        SequenceStudioModel([]),
        StudioStore(tmp_path / "context-budget.sqlite3"),
        max_context_tokens=3_000,
    )
    context: dict[str, object] = {
        "files": [f"src/module_{index}.py" for index in range(240)],
        "messages": [{"role": "user", "content": "x" * 8_000} for _ in range(12)],
        "recent_observations": [{"payload": "y" * 8_000} for _ in range(8)],
        "historical_summaries": ["z" * 1_000 for _ in range(16)],
        "retrieved_context": {
            "symbols": [{"name": f"symbol_{index}"} for index in range(30)],
            "snippets": [{"content": "code" * 2_000} for _ in range(8)],
        },
    }

    fitted, estimated, trimmed = agent._fit_context(context)

    assert estimated <= 3_100
    assert fitted["context_budget"]["limit_tokens"] == 3_000
    assert "retrieved_snippets" in trimmed
    assert "older_observations" in trimmed
    assert "older_messages" in trimmed


def test_long_history_reduction_is_recorded_as_context_compression(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "context-compression.sqlite3")
    session = StudioSession(
        session_id="context-compression",
        repo_root=str(tmp_path),
        provider="openai",
        model="gpt-test",
        reasoning_effort="low",
        messages=[
            StudioMessage(
                role="user" if index % 2 == 0 else "assistant",
                content=f"historical message {index}: " + "x" * 1_000,
            )
            for index in range(24)
        ],
    )
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.RESPOND,
                rationale="answer current request",
                message="done",
            )
        ]
    )

    asyncio.run(
        StudioAgent(model, store, max_context_tokens=4_000).handle(
            session, "What is the current state?"
        )
    )

    events = store.events(session.session_id)
    compression = next(event for event in events if event["event_type"] == "context_compressed")
    assert "older_messages" in compression["payload"]["trimmed"]
    assert compression["payload"]["before_tokens"] > compression["payload"]["after_tokens"]
    assert compression["payload"]["after_tokens"] <= 4_000


def test_model_observation_keeps_audit_payload_out_of_prompt() -> None:
    source = "print('before')\n" * 2_000
    updated = "print('after')\n" * 2_000
    observation = StudioObservation(
        kind="edit",
        summary="已修改 calculator.py。",
        payload={
            "path": "calculator.py",
            "intent": "修复计算逻辑",
            "before": source,
            "after": updated,
            "diff": f"-{source}+{updated}",
            "content_sha256": "abc123",
        },
    )

    projected = StudioAgent._compact_observation(observation)

    assert projected["payload"] == {
        "path": "calculator.py",
        "intent": "修复计算逻辑",
        "content_sha256": "abc123",
    }
    assert observation.payload["before"] == source
    assert observation.payload["after"] == updated
    assert "diff" in observation.payload


def test_layered_context_leaves_output_headroom_after_large_tool_history(tmp_path: Path) -> None:
    agent = StudioAgent(
        SequenceStudioModel([]),
        StudioStore(tmp_path / "layered-context.sqlite3"),
        max_context_tokens=16_000,
    )
    observations = [
        StudioObservation(
            kind="edit",
            summary=f"已修改 module_{index}.py。",
            payload={
                "path": f"module_{index}.py",
                "before": "a" * 40_000,
                "after": "b" * 40_000,
                "diff": "c" * 80_000,
                "content_sha256": str(index),
            },
        )
        for index in range(8)
    ]
    context: dict[str, object] = {
        "current_request": "继续完成计算器并验证结果",
        "recent_observations": [
            agent._compact_observation(observation) for observation in observations
        ],
        "messages": [{"role": "user", "content": "历史消息" * 3_000}] * 8,
        "files": [f"module_{index}.py" for index in range(200)],
    }

    fitted, estimated, trimmed = agent._fit_context(context)

    assert estimated <= 16_000
    assert fitted["context_budget"]["target_tokens"] == 12_800
    assert fitted["context_budget"]["reserved_tokens"] == 3_200
    assert "older_messages" in trimmed
    serialized = json.dumps(fitted, ensure_ascii=False)
    assert "print('before')" not in serialized
    assert '"diff"' not in serialized


def test_auto_mode_requires_verification_after_code_change(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    model = SequenceStudioModel(
        [
            StudioDecision(
                action=StudioAction.EDIT,
                rationale="修复实现",
                path="calc.py",
                old_text="return a - b",
                new_text="return a + b",
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="尝试提前完成",
                message="修改完成。",
            ),
            StudioDecision(
                action=StudioAction.RUN_TESTS,
                rationale="补充验证证据",
                command=["python", "-m", "pytest", "-q"],
            ),
            StudioDecision(
                action=StudioAction.FINISH,
                rationale="验证与自审完成",
                message="修复已经通过自动验证。",
            ),
        ]
    )
    store = StudioStore(tmp_path / "auto-gate.sqlite3")
    session = StudioSession(
        session_id="auto-gate-1",
        repo_root=str(repository),
        provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
    )

    result = asyncio.run(StudioAgent(model, store).handle(session, "修复并验证加法"))

    assert result.status == "completed"
    assert result.verification_passed is True
    assert [item.kind for item in result.observations] == [
        "edit",
        "verification_gate",
        "test",
        "result_review",
        "final_review",
    ]


def test_studio_api_creates_project_files_and_empty_folders(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    client = TestClient(create_app(Settings(database_path=tmp_path / "entries.sqlite3")))
    response = client.post(
        "/studio-api/sessions",
        json={
            "repo_root": str(repository),
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
        },
    )
    session_id = response.json()["session_id"]

    folder = client.post(
        f"/studio-api/sessions/{session_id}/files",
        json={"kind": "folder", "path": "src/components"},
    )
    file = client.post(
        f"/studio-api/sessions/{session_id}/files",
        json={"kind": "file", "path": "src/app.py"},
    )

    assert folder.status_code == 201
    assert file.status_code == 201
    assert (repository / "src" / "components").is_dir()
    assert (repository / "src" / "app.py").read_text(encoding="utf-8") == ""
    tree = client.get(f"/studio-api/sessions/{session_id}/files").json()
    assert "src/app.py" in tree["files"]
    assert "src/components" in tree["directories"]
    assert client.post(
        f"/studio-api/sessions/{session_id}/files",
        json={"kind": "file", "path": "../outside.py"},
    ).status_code == 400


def test_studio_api_serves_workspace_and_validates_models(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    client = TestClient(create_app(Settings(database_path=tmp_path / "api.sqlite3")))
    page = client.get("/studio")
    assert page.status_code == 200
    assert "<title>RAgent</title>" in page.text
    assert "brand-logo" not in page.text
    assert '<span class="brand-name">RAgent</span>' not in page.text
    assert "通用编码 Agent" not in page.text
    brand = client.get("/studio-brand")
    assert brand.status_code == 200
    assert brand.headers["content-type"] == "image/png"
    asset_path = re.search(r'src="(/assets/[^"]+\.js)"', page.text)
    assert asset_path is not None
    asset = client.get(asset_path.group(1))
    assert asset.status_code == 200
    assert "新建对话" in asset.text
    assert "选择文件夹" in asset.text
    assert "API 配置" in asset.text
    assert "让 RAgent 阅读代码、修改文件，或检查测试结果。" in asset.text
    assert client.get("/assets/../index.html").status_code == 404
    for contract in (
        "/system/directories",
        "/file-change?path=",
        "/provider-models",
        "更改前",
        "更改后",
        'name:"verification_mode"',
        'name:"test_command"',
    ):
        assert contract in asset.text
    assert "会话设置" in asset.text
    assert "/settings" in asset.text
    assert "保存失败：" in asset.text
    assert "保存中…" in asset.text
    response = client.post(
        "/studio-api/sessions",
        json={
            "repo_root": str(repository),
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
        },
    )
    assert response.status_code == 201
    session_id = response.json()["session_id"]
    session_response = client.get(f"/studio-api/sessions/{session_id}")
    assert session_response.status_code == 200
    assert session_response.json()["verification_mode"] == "auto"
    assert session_response.json()["response_style"] == "concise"
    assert session_response.json()["context_limit_tokens"] == 1_000_000
    files = client.get(f"/studio-api/sessions/{session_id}/files").json()["files"]
    assert files == ["calc.py", "test_calc.py"]
    project = client.get(f"/studio-api/sessions/{session_id}/project").json()
    assert project == {"stacks": ["Unknown"], "markers": []}
    unchanged = client.get(
        f"/studio-api/sessions/{session_id}/file-change", params={"path": "calc.py"}
    )
    assert unchanged.status_code == 200
    assert unchanged.json()["changed"] is False
    assert "return a - b" in unchanged.json()["current"]
    updated = client.patch(
        f"/studio-api/sessions/{session_id}/settings",
        json={
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "reasoning_effort": "low",
            "response_style": "teaching",
            "verification_mode": "quick",
            "test_command": "python -m pytest -q",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["model"] == "deepseek-v4-pro"
    assert updated.json()["context_limit_tokens"] == 1_000_000
    assert updated.json()["verification_mode"] == "quick"
    assert updated.json()["response_style"] == "teaching"
    inherited = client.post(
        "/studio-api/sessions",
        json={
            "repo_root": str(repository),
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
        },
    )
    inherited_session = client.get(f"/studio-api/sessions/{inherited.json()['session_id']}")
    assert inherited_session.json()["response_style"] == "teaching"
    official_model = client.patch(
        f"/studio-api/sessions/{session_id}/settings",
        json={
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "verification_mode": "quick",
            "test_command": "python -m pytest -q",
        },
    )
    assert official_model.status_code == 200
    assert official_model.json()["context_limit_tokens"] == 1_050_000
    proxy_model = client.patch(
        f"/studio-api/sessions/{session_id}/settings",
        json={
            "provider": "openai",
            "model": "codex-auto-review",
            "reasoning_effort": "medium",
            "verification_mode": "quick",
            "test_command": "python -m pytest -q",
        },
    )
    assert proxy_model.status_code == 200
    assert proxy_model.json()["provider"] == "openai"
    assert proxy_model.json()["model"] == "codex-auto-review"
    assert proxy_model.json()["context_limit_tokens"] == 16_000
    invalid_proxy_model = client.patch(
        f"/studio-api/sessions/{session_id}/settings",
        json={
            "provider": "openai",
            "model": "bad model; remove-files",
            "reasoning_effort": "medium",
            "verification_mode": "quick",
            "test_command": "python -m pytest -q",
        },
    )
    assert invalid_proxy_model.status_code == 400
    studio_store = StudioStore(tmp_path / "api.sqlite3")
    modified = studio_store.load(session_id)
    assert modified is not None
    modified.changed_files = ["calc.py"]
    studio_store.save(modified, "test_modified", {})
    strict_after_edit = client.patch(
        f"/studio-api/sessions/{session_id}/settings",
        json={
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "reasoning_effort": "low",
            "verification_mode": "strict",
            "test_command": "python -m pytest -q",
        },
    )
    assert strict_after_edit.status_code == 409
    assert (
        client.get(
            f"/studio-api/sessions/{session_id}/file", params={"path": "../secret.txt"}
        ).status_code
        == 400
    )
    invalid = client.post(
        "/studio-api/sessions",
        json={
            "repo_root": str(repository),
            "provider": "deepseek",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
        },
    )
    assert invalid.status_code == 400
    unsafe_strict = client.post(
        "/studio-api/sessions",
        json={
            "repo_root": str(repository),
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
            "verification_mode": "strict",
            "test_command": "powershell Remove-Item calc.py",
        },
    )
    assert unsafe_strict.status_code == 400
    assert client.delete(f"/studio-api/sessions/{session_id}").status_code == 204
    assert client.get(f"/studio-api/sessions/{session_id}").status_code == 404


def test_frozen_project_root_uses_pyinstaller_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("veripatch.api.sys.frozen", True, raising=False)
    monkeypatch.setattr("veripatch.api.sys._MEIPASS", str(tmp_path), raising=False)
    assert _project_root() == tmp_path


def test_upgrade_acceptance_checks_run_independently(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(database_path=tmp_path / "acceptance.sqlite3")))
    catalog = client.get("/studio-api/upgrade-checks")
    assert catalog.status_code == 200
    assert catalog.json()["version"] == "3.0.0"
    assert len(catalog.json()["checks"]) == 5

    single = client.post("/studio-api/upgrade-checks/task_contract")
    assert single.status_code == 200
    assert single.json()["status"] == "passed"
    assert single.json()["evidence"]

    all_checks = client.post("/studio-api/upgrade-checks/all")
    assert all_checks.status_code == 200
    assert {item["status"] for item in all_checks.json()["results"]} == {"passed"}
    assert client.post("/studio-api/upgrade-checks/missing").status_code == 404
