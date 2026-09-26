import asyncio

import pytest

from veripatch.config import Settings
from veripatch.domain import IssueSpec
from veripatch.models.base import ModelContext
from veripatch.models.scripted import DiscountBugDemoModel
from veripatch.providers import create_model


def test_provider_registry_builds_scripted_model() -> None:
    assert isinstance(create_model("scripted-demo", Settings()), DiscountBugDemoModel)


def test_scripted_demo_explains_decisions_in_chinese() -> None:
    context = ModelContext(
        issue=IssueSpec(issue_id="demo", title="折扣错误", description="修复折扣计算"),
        step=0,
        changed_files=[],
        index_summary={},
        recent_observations=[],
    )
    reply = asyncio.run(DiscountBugDemoModel().decide(context))
    assert reply.decision.rationale == "定位失败行为中提到的函数。"


def test_provider_registry_rejects_unknown_and_retired_deepseek_names() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        create_model("other", Settings())
    with pytest.raises(ValueError, match="Retired"):
        create_model("deepseek", Settings(deepseek_model="deepseek-chat"))
