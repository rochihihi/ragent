"""Neutral model-led turn policy and remaining request helpers for Studio."""

from __future__ import annotations

import re
from dataclasses import dataclass

from veripatch.studio_domain import StudioAction



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


def model_intent_policy() -> IntentPolicy:
    """Neutral context for model understanding; action safety is checked at execution."""
    return IntentPolicy(
        intent="unresolved", confidence="unknown",
        rationale="由模型结合当前请求和会话上下文理解任务",
        allowed_actions=frozenset(StudioAction),
    )


_CONTEXTUAL_CONTINUATIONS = {
    "接着做",
    "继续处理",
    "按刚才的继续",
    "照刚才说的继续",
    "继续刚才的任务",
    "resume that",
    "keep going",
}


def is_contextual_continuation(message: str) -> bool:
    """Recognize short follow-ups that inherit an unfinished contract."""
    return message.strip().casefold() in _CONTEXTUAL_CONTINUATIONS




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
