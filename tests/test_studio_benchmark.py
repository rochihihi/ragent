import asyncio

from scripts import eval_flash_dialogue as benchmark
from veripatch.studio_domain import StudioDecision, StudioReply


def test_benchmark_rejects_false_completion(monkeypatch):
    async def lying_model(self, context):
        return StudioReply(
            decision=StudioDecision(action="respond", rationale="done", message="已创建 a.txt。")
        )

    monkeypatch.setattr(benchmark.MeasuredModel, "decide", lying_model)
    result = asyncio.run(benchmark.run_case("create"))
    assert not result["passed"]
    assert not result["checks"]["files"]


def test_benchmark_multiturn_uses_real_runtime():
    result = asyncio.run(benchmark.run_case("multi_turn"))
    assert result["passed"]
    assert result["steps"] == 4


def test_fixed_suite_covers_recent_regressions():
    assert {"idempotent", "linked_test", "bulk_delete"} <= set(benchmark.CASES)
    for name in ("idempotent", "linked_test", "bulk_delete"):
        result = asyncio.run(benchmark.run_case(name))
        assert result["passed"], result
