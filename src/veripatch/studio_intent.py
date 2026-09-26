"""Deterministic turn intent and capability policy for Studio."""

from __future__ import annotations

import re
from dataclasses import dataclass

from veripatch.studio_domain import SemanticIntentAssessment, StudioAction, StudioTaskContract

READ_ACTIONS = {
    StudioAction.LIST_FILES,
    StudioAction.SEARCH,
    StudioAction.READ,
    StudioAction.GIT_STATUS,
    StudioAction.GIT_DIFF,
    StudioAction.GIT_LOG,
    StudioAction.GIT_BRANCH,
    StudioAction.MCP_CALL,
    StudioAction.RESPOND,
    StudioAction.FINISH,
    StudioAction.FAIL,
}
WRITE_ACTIONS = {
    StudioAction.EDIT,
    StudioAction.APPLY_PATCH,
    StudioAction.CREATE,
    StudioAction.MOVE_FILE,
    StudioAction.COPY_FILE,
    StudioAction.DELETE_PATH,
}
COMMAND_ACTIONS = {
    StudioAction.RUN_TESTS,
    StudioAction.RUN_COMMAND,
    StudioAction.START_TERMINAL,
    StudioAction.POLL_TERMINAL,
    StudioAction.WRITE_TERMINAL,
    StudioAction.STOP_TERMINAL,
    StudioAction.INSPECT_PROCESSES,
}
GIT_WRITE_ACTIONS = {StudioAction.GIT_COMMIT, StudioAction.GIT_RESTORE}


@dataclass(frozen=True)
class IntentPolicy:
    intent: str
    confidence: str
    rationale: str
    allowed_actions: frozenset[StudioAction]
    mutation_requested: bool = False
    verification_requested: bool = False
    launch_requested: bool = False
    denied_actions: frozenset[StudioAction] = frozenset()
    evidence_required: bool = False


_CONTEXTUAL_CONTINUATIONS = {
    "接着做",
    "继续处理",
    "按刚才的继续",
    "照刚才说的继续",
    "继续刚才的任务",
    "resume that",
    "keep going",
}


def semantic_policy(fallback: IntentPolicy, assessment: SemanticIntentAssessment) -> IntentPolicy:
    """Compile semantic intent into capabilities; lexical misses are not denials."""
    if (
        assessment.requires_clarification
        and not assessment.clarification_question
        and not assessment.requested_actions
        and fallback.mutation_requested
    ):
        # An empty uncertainty result carries no usable correction to an explicit request.
        assessment.requires_clarification = False
        return fallback
    known = {action.value: action for action in StudioAction}
    requested = {known[name] for name in assessment.requested_actions if name in known}
    # A conversational workflow can be labelled 'execute' in ordinary language.
    # Concrete response-only actions carry no filesystem/command effects.
    if (
        assessment.requested_actions
        and all(name in known for name in assessment.requested_actions)
        and requested <= {StudioAction.RESPOND, StudioAction.FINISH}
        and not fallback.mutation_requested
        and not fallback.verification_requested
        and not fallback.launch_requested
    ):
        assessment.intent = "answer"
    denied = set(fallback.denied_actions) | {
        known[name] for name in assessment.prohibited_actions if name in known
    }
    allowed = set(READ_ACTIONS) | {StudioAction.REQUEST_PERMISSION}
    effect_intent = assessment.intent not in {"answer", "analysis"}
    # An unsupported side-effect label is not enough to grant capabilities.
    concrete = bool(requested & (WRITE_ACTIONS | COMMAND_ACTIONS | GIT_WRITE_ACTIONS))
    if (
        effect_intent
        and not concrete
        and not (
            fallback.mutation_requested
            or fallback.verification_requested
            or fallback.launch_requested
        )
    ):
        assessment.requires_clarification = True
        assessment.clarification_question = (
            assessment.clarification_question or "你希望具体执行什么操作？"
        )
    if effect_intent and not assessment.requires_clarification:
        allowed.update(requested)
        if assessment.intent == "change":
            allowed.update(WRITE_ACTIONS | COMMAND_ACTIONS)
        elif assessment.intent in {"verify", "execute", "install", "launch_only"}:
            allowed.update(COMMAND_ACTIONS)
    allowed.difference_update(denied)
    return IntentPolicy(
        intent=assessment.intent,
        confidence=assessment.confidence,
        rationale=assessment.rationale,
        allowed_actions=frozenset(allowed),
        denied_actions=frozenset(denied),
        mutation_requested=effect_intent and bool(allowed & WRITE_ACTIONS),
        verification_requested=effect_intent and bool(requested & {StudioAction.RUN_TESTS}),
        launch_requested=assessment.intent == "launch_only" and bool(allowed & COMMAND_ACTIONS),
        evidence_required=fallback.evidence_required or bool(assessment.questions),
    )


def is_contextual_continuation(message: str) -> bool:
    """Recognize short follow-ups that inherit an unfinished contract."""
    return message.strip().casefold() in _CONTEXTUAL_CONTINUATIONS


def ambiguous_side_effect_request(message: str) -> bool:
    """Return true when prose hints at action without authorizing a concrete effect."""
    normalized = message.strip().casefold()
    if requests_code_change(normalized):
        return False
    return bool(
        re.search(
            r"(?i)(?:就这么办|照(?:刚才|上面).{0,8}(?:办|处理)|"
            r"(?:弄|搞|收拾|处理)一下|先看看.{0,16}(?:不行|有问题).{0,8}(?:处理|改))",
            normalized,
        )
    )


def _without_negated_mutations(message: str) -> str:
    normalized = re.sub(
        r"(?i)(?:不要|不得|无需|不需要|不允许|禁止|停止|暂停|不是让你|不是要你|别|不)\s*"
        r"[^，。；;\n]{0,24}?"
        r"(?:修改|改动|改|编辑|写入|删除|创建|新建|创作)[^，。；;\n]*",
        "",
        message,
    )
    return re.sub(
        r"(?i)\b(?:do\s+not|don't|must\s+not|without)\s+"
        r"(?:modify|edit|change|write|delete|create)[^,.;\n]*",
        "",
        normalized,
    )


def requests_code_change(message: str) -> bool:
    normalized = _without_negated_mutations(message)
    normalized = re.sub(r"(?:修改|改动|改)(?:了|过)(?:什么|哪些|哪几个)", "", normalized)
    normalized = re.sub(r"(?i)(?:怎么|如何)\s*(?:做|实现|处理)", "", normalized)
    normalized = re.sub(r"(?i)(?:现有|当前|这两个|这个|该)\s*实现(?:的)?", "", normalized)
    deliberative_question = bool(
        re.search(
            r"(?i)(?:你觉得|你认为|是否|是不是|该不该|要不要|会不会|值不值得|"
            r"怎么|如何|should\s+(?:we|i)|do\s+you\s+think)",
            normalized,
        )
    )
    explicit_imperative = bool(
        re.search(
            r"(?i)(?:请|帮我|直接|现在|立即|马上|务必|给我).{0,20}"
            r"(?:改|修改|调整|重构|修复|创建|删除|实现|优化)|"
            r"\b(?:please|go ahead and)\b",
            normalized,
        )
    )
    if deliberative_question and not explicit_imperative:
        return False
    normalized = re.sub(
        r"(?i)(?:(?:有哪些|哪些|有什么|可以|可|如何|怎么|未来|值得)\s*"
        r"(?:改进|优化)(?:的)?(?:地方|之处|方向|建议|点)?|"
        r"(?:改进|优化)(?:建议|方向|空间|点|之处))",
        "",
        normalized,
    )
    normalized = re.sub(
        r"(?i)\b(?:improvement|optimization)\s+(?:ideas?|suggestions?|areas?)\b",
        "",
        normalized,
    )
    return bool(
        re.search(
            r"(?i)(?:换(?:成|种|个)?(?:语言)?|改成|改用|"
            r"(?:重新)?(?:弄|做)(?:个|一个|份)?(?:新)?"
            r"(?:网页|页面|网站|文件|应用|程序|项目|功能|组件|按钮|界面)|"
            r"改(?!完|好|过|后|进)|改进|重写|重新写|编写|创建|新建|创作|制作|"
            r"写(?!完|好|过)|做(?!了?什么|完|好|过)|生成|转换|修复|优化|升级|"
            r"添加|加一行|新增|删除|重命名|复制|拷贝|移动|"
            r"留(?:个|一份)?副本|备份|实现|开发|迁移|重构|调整|完善|美化|"
            r"修改(?!后|完|好|过)|\b(?:refactor|modify|edit|create|delete|implement)\b)",
            normalized,
        )
    )


def contradictory_verification_request(message: str) -> bool:
    """Detect an explicit request to verify combined with a blanket command prohibition."""
    command_denied = bool(
        re.search(
            r"(?i)(?:不要|不得|不允许|禁止|别)运行(?:任何|任何的)?命令"
            r"|\b(?:do\s+not|don't|must\s+not)\s+(?:run|execute)\s+(?:any\s+)?commands?\b",
            message,
        )
    )
    without_denials = re.sub(
        r"(?i)(?:不要|不得|不允许|禁止|别)运行(?:任何|任何的)?命令"
        r"|\b(?:do\s+not|don't|must\s+not)\s+(?:run|execute)\s+(?:any\s+)?commands?\b",
        "",
        message,
    )
    verification_status_question = bool(
        re.search(
            r"(?i)(?:告诉我|说明|回答|查看|确认)?(?:最终|最后)?(?:是否|有没有|有无)"
            r".{0,12}(?:通过|运行|执行).{0,8}(?:测试|验证|检查)?|"
            r"(?:测试|验证|检查)(?:是否|有没有|有无).{0,8}(?:通过|运行|执行)",
            without_denials,
        )
    )
    verification_requested = not verification_status_question and bool(
        re.search(
            r"(?i)(?:运行|执行)\s*(?:一下|全部|所有)?\s*"
            r"(?:测试|验证|\bpytest\b|\btests?\b(?![/_])|\blint\b|\bbuild\b)",
            without_denials,
        )
    )
    return command_denied and verification_requested


def classify_intent(message: str) -> IntentPolicy:
    mutation_requested = requests_code_change(message)
    mutation_denied = (
        _without_negated_mutations(message) != message and not mutation_requested
    )
    command_denied = bool(
        re.search(
            r"(?i)(?:不要|不得|无需|不需要|不允许|禁止|别|不)\s*"
            r"(?:运行|执行|测试|验证|构建|启动|安装)"
            r"|\b(?:do\s+not|don't|must\s+not|without)\s+"
            r"(?:run|execute|test|verify|build|launch|install)\b",
            message,
        )
    )
    mutation = mutation_requested and not mutation_denied
    asks_execution_method = bool(
        re.search(r"(?i)(?:怎么|如何).{0,12}(?:测试|验证|检查|运行|启动|构建|安装)", message)
    )
    asks_verification_status = bool(
        re.search(
            r"(?i)(?:告诉我|说明|回答|查看|确认)?(?:最终|最后)?(?:是否|有没有|有无)"
            r".{0,12}(?:通过|运行|执行).{0,8}(?:测试|验证|检查)?|"
            r"(?:测试|验证|检查)(?:是否|有没有|有无).{0,8}(?:通过|运行|执行)",
            message,
        )
    )
    asks_verification_status = asks_verification_status or bool(
        re.search(r"(?:测试|验证|检查).{0,8}(?:了吗|了么|过吗|过么|没)[？?]?", message)
    )
    launch = (
        not command_denied
        and not asks_execution_method
        and bool(
            re.search(
                r"(?i)(?:请|帮我|给我|再|重新)?(?:打开|启动|运行起来)|\b(?:open|launch)\b", message
            )
        )
    )
    verification = (
        not command_denied
        and not asks_execution_method
        and not asks_verification_status
        and bool(
            re.search(
                r"(?i)(?:验证|测试|复现.{0,20}(?:失败|错误)|排查.{0,20}(?:失败|错误)|"
                r"检查.{0,40}(?:语法|编译|构建|测试)|运行\s+(?:python|pytest|go|npm)|"
                r"\b(?:run\s+)?(?:pytest|verify|test|lint|build|compile)\b)",
                message,
            )
        )
    )
    install = not command_denied and bool(
        re.search(r"(?i)(?:安装|配置).{0,20}(?:依赖|环境|软件|工具链|包)|\binstall\b", message)
    )
    git_write = not mutation_denied and bool(
        re.search(r"(?i)(?:git\s*)?(?:提交|回滚|撤销提交)|\b(?:commit|revert)\b", message)
    )
    execute = (
        not command_denied
        and not asks_execution_method
        and bool(
            re.search(
                r"(?i)(?:执行|运行).{0,20}(?:命令|脚本|程序|项目)|\b(?:execute|run)\b",
                message,
            )
        )
    )
    analysis = bool(
        re.search(
            r"(?i)(?:只(?:读取并|读并)?分析|仅分析|全面介绍|介绍.*项目|项目结构|代码结构|文件作用|"
            r"(?:读取|读一下|查看).{0,80}?[\w./\\-]+\."
            r"(?:py|js|jsx|ts|tsx|html|css|go|rs|java|cs|cpp|c|json|ya?ml|txt|md|csv)|"
            r"优点|缺点|分析.*(?:结构|项目|代码)|介绍.*(?:文件|架构|代码)|"
            r"改进建议|优化建议|可以改进|可以优化|潜在改进|如何改进|"
            r"overview|architecture|improvement suggestions?)",
            message,
        )
    )

    allowed = set(READ_ACTIONS)
    denied: set[StudioAction] = set()
    if mutation_denied:
        denied.update(WRITE_ACTIONS | GIT_WRITE_ACTIONS)
    if command_denied:
        denied.update(COMMAND_ACTIONS)
    reasons: list[str] = []
    if mutation:
        allowed.update(WRITE_ACTIONS | COMMAND_ACTIONS)
        reasons.append("检测到明确的文件修改请求")
    if verification or install or launch or execute:
        allowed.update(COMMAND_ACTIONS)
        reasons.append("检测到验证、安装或启动请求")
    if git_write:
        allowed.update(GIT_WRITE_ACTIONS)
        reasons.append("检测到明确的 Git 写操作")
    # Permission requests are allowed for external reads and for explicitly
    # authorized writes/commands; the permission UI remains the final gate.
    allowed.add(StudioAction.REQUEST_PERMISSION)
    allowed.difference_update(denied)
    if mutation or verification or install or launch or execute or git_write:
        if mutation:
            intent = "change"
        elif launch:
            intent = "launch_only"
        elif verification:
            intent = "verify"
        elif install:
            intent = "install"
        else:
            intent = "execute"
        confidence = "high"
    elif analysis:
        intent = "analysis"
        confidence = "high"
        reasons.append("检测到项目介绍、评估或建议语义")
    else:
        intent = "answer"
        confidence = "medium"
        reasons.append("未检测到明确副作用，默认只读")
    return IntentPolicy(
        intent=intent,
        confidence=confidence,
        rationale="；".join(reasons),
        allowed_actions=frozenset(allowed),
        mutation_requested=mutation,
        verification_requested=verification,
        launch_requested=launch,
        denied_actions=frozenset(denied),
        evidence_required=bool(
            re.search(
                r"(?i)(?:是否|有没有|什么时候|几点|哪些|做了什么|"
                r"修改了|运行了|测试结果|上下文|压缩|状态|记录|"
                r"when|whether|what changed|test result|context|compress|status|history)",
                message,
            )
        ),
    )


def denied_action_reason(contract: StudioTaskContract, action: StudioAction) -> str | None:
    if not contract.allowed_actions:
        return None
    if action is StudioAction.BATCH:
        return None
    if action.value in contract.allowed_actions:
        return None
    return (
        f"当前任务意图为 {contract.intent}，未授权 {action.value}。"
        "只能使用任务契约中的 allowed_actions。"
    )
