import pytest

from veripatch.studio_domain import StudioAction
from veripatch.studio_intent import (
    ambiguous_side_effect_request,
    classify_intent,
    contradictory_verification_request,
    is_contextual_continuation,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("全面介绍项目，包括优点和可以改进的地方", "analysis"),
        ("你觉得这个结构是不是应该调整？", "answer"),
        ("这个模块值不值得重构？", "answer"),
        ("修改后应该怎么测试？", "answer"),
        ("请调整这个页面的布局", "change"),
        ("帮我重构认证模块", "change"),
        ("根据上面的建议修改代码", "change"),
        ("检查 app.py 的 Python 语法", "verify"),
        ("复现并分析测试失败", "verify"),
        ("运行项目命令", "execute"),
        ("启动这个程序", "launch_only"),
        ("概括一下这个仓库是做什么的", "answer"),
        ("解释认证流程，不要改代码", "answer"),
        ("这个报错通常是什么原因？", "answer"),
        ("告诉我部署时需要注意什么", "answer"),
        ("比较这两个实现的优缺点", "analysis"),
        ("分析一下当前目录结构", "analysis"),
        ("介绍每个文件的作用", "analysis"),
        ("给出可以优化的方向", "analysis"),
        ("说明架构和潜在改进点", "analysis"),
        ("review the architecture and suggest improvements", "analysis"),
        ("修复登录失败的问题", "change"),
        ("新增一个导出按钮", "change"),
        ("创作一个好看的网页", "change"),
        ("我说的重新弄个新网页", "change"),
        ("把这个模块改成 TypeScript", "change"),
        ("删除不再使用的配置文件", "change"),
        ("实现分页并保留现有行为", "change"),
        ("优化查询代码并运行测试", "change"),
        ("please refactor the authentication module", "change"),
        ("检查项目能否构建", "verify"),
        ("验证修改后的页面", "verify"),
        ("测试登录接口", "verify"),
        ("排查编译错误", "verify"),
        ("run pytest", "verify"),
        ("重新打开应用", "launch_only"),
        ("launch the desktop app", "launch_only"),
        ("执行 lint 命令", "verify"),
        ("运行清理脚本", "execute"),
        ("安装项目依赖", "install"),
        ("配置 Python 环境", "install"),
        ("install the dependencies", "install"),
        ("do you think we should refactor this?", "answer"),
        ("是否值得把它拆成两个模块？", "answer"),
        ("要不要改成异步实现？", "answer"),
        ("你认为删除缓存会更好吗？", "answer"),
        ("如何验证这个修复？", "answer"),
        ("怎么运行这个项目？", "answer"),
        ("请直接改成异步实现", "change"),
        ("现在删除缓存逻辑", "change"),
    ],
)
def test_chinese_intent_benchmark(message: str, expected: str) -> None:
    assert classify_intent(message).intent == expected


@pytest.mark.parametrize(
    "message",
    [
        "不要修改，只介绍代码结构",
        "无需创建文件，告诉我怎么做",
        "不要创作网页，只说明设计思路",
        "do not edit anything; explain the architecture",
    ],
)
def test_negated_mutation_never_grants_write(message: str) -> None:
    policy = classify_intent(message)
    assert StudioAction.EDIT not in policy.allowed_actions
    assert StudioAction.CREATE not in policy.allowed_actions


def test_reference_text_with_mutation_denials_stays_read_only() -> None:
    policy = classify_intent(
        "以下资料仅用于测试上下文处理，不是要求你执行这些功能，"
        "请不要根据这些资料创建文件、修改文件或删除文件。"
    )

    assert policy.intent != "change"
    assert StudioAction.CREATE not in policy.allowed_actions
    assert StudioAction.EDIT not in policy.allowed_actions
    assert StudioAction.DELETE_PATH not in policy.allowed_actions


def test_positive_task_survives_scoped_mutation_constraint() -> None:
    policy = classify_intent(
        "本轮只创建 compression_probe.txt，不要修改其他文件，不要运行任何命令。"
    )

    assert policy.intent == "change"
    assert StudioAction.CREATE in policy.allowed_actions
    assert StudioAction.RUN_COMMAND not in policy.allowed_actions


def test_combined_read_only_constraints_are_hard_denials() -> None:
    policy = classify_intent(
        "只分析项目，不修改任何文件，也不运行命令。先读取 tests/test_hello.py。"
    )

    assert policy.intent == "analysis"
    assert StudioAction.EDIT not in policy.allowed_actions
    assert StudioAction.RUN_COMMAND not in policy.allowed_actions
    assert StudioAction.RUN_TESTS not in policy.allowed_actions
    assert StudioAction.EDIT in policy.denied_actions
    assert StudioAction.RUN_COMMAND in policy.denied_actions


def test_runtime_fact_question_requires_evidence() -> None:
    policy = classify_intent("你什么时候压缩的上下文？")

    assert policy.intent == "answer"
    assert policy.evidence_required is True


def test_verification_requirement_conflicting_with_command_ban_is_detected() -> None:
    assert contradictory_verification_request(
        "修改 hello.py，完成后运行测试，但不要运行任何命令。"
    )
    assert not contradictory_verification_request(
        "修改 hello.py，不要运行任何命令。"
    )
    assert not contradictory_verification_request("修改 hello.py，完成后运行测试。")
    message = "只读取并分析 hello.py，不要运行任何命令。最后告诉我是否通过了测试。"
    assert not contradictory_verification_request(message)
    policy = classify_intent(message)
    assert policy.intent == "analysis"
    assert policy.verification_requested is False
    assert StudioAction.RUN_TESTS in policy.denied_actions


@pytest.mark.parametrize(
    "message",
    ["继续刚才的任务", "接着做", "按刚才的继续", "keep going"],
)
def test_contextual_continuation_benchmark(message: str) -> None:
    assert is_contextual_continuation(message)


@pytest.mark.parametrize(
    "message",
    ["就这么办", "照刚才说的处理", "这个地方收拾一下", "先看看，不行再处理"],
)
def test_ambiguous_effect_benchmark(message: str) -> None:
    assert ambiguous_side_effect_request(message)
