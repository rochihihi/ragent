"""Deterministic offline model used only by the frozen demonstration task."""

from __future__ import annotations

from veripatch.domain import ActionKind, AgentDecision, FileEdit
from veripatch.models.base import ModelContext, ModelReply


class DiscountBugDemoModel:
    """A predictable policy that exercises the complete runtime without an API key."""

    async def decide(self, context: ModelContext) -> ModelReply:
        kinds = {observation.kind for observation in context.recent_observations}
        if "search" not in kinds:
            decision = AgentDecision(
                action=ActionKind.SEARCH,
                rationale="定位失败行为中提到的函数。",
                query="calculate_discount",
            )
        elif "read" not in kinds:
            decision = AgentDecision(
                action=ActionKind.READ,
                rationale="修改之前先阅读折扣功能的具体实现。",
                path="discount/calc.py",
                start_line=1,
                end_line=120,
            )
        elif "edit" not in kinds:
            decision = AgentDecision(
                action=ActionKind.EDIT,
                rationale=("当前实现直接减去了整个百分比数值，没有先将百分比转换为小数。"),
                edits=[
                    FileEdit(
                        path="discount/calc.py",
                        old_text="return price * (1 - percent)",
                        new_text="return price * (1 - percent / 100)",
                    )
                ],
            )
        else:
            decision = AgentDecision(
                action=ActionKind.RUN_TESTS,
                rationale="使用不可篡改的测试命令验证修复结果。",
            )
        return ModelReply(decision=decision)
