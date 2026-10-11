import asyncio
import json
import threading
import time

import pytest

from veripatch.embedding_assets import model_directory, verify_artifacts
from veripatch.studio_domain import StudioSession
from veripatch.studio_embeddings import LocalEncoder, SemanticIndex, retrieve_memory
from veripatch.studio_memory import (
    bm25_scores,
    forget_model_note,
    fused_scores,
    project_memory,
    remember,
)


def session(tmp_path, name="test"):
    return StudioSession(session_id=name, repo_root=str(tmp_path), provider="deepseek",
                         model="test", reasoning_effort="low")


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[1., 0.] if text in {"登录凭据存放在哪里", "令牌保存于保险库"} else [0., 1.]
                for text in texts]


def test_bm25_identifier_chinese_frequency_and_length():
    scores = bm25_scores(["src/billing.py 退款", "billing.py billing.py", "天气晴朗"], "billing.py")
    assert scores[1] > scores[0] > scores[2] == 0
    assert bm25_scores(["检查退款金额", "查看日志"], "退款")[0] > 0
    assert bm25_scores(["长文 " * 200 + "退款", "退款"], "退款")[1] > 1
    assert bm25_scores(["", "nothing"], "") == [0., 0.]


def test_rrf_uses_ranks_and_excludes_zero_hits():
    scores = fused_scores(["billing.py", "退款计算", "天气"], "billing.py", [.6, .9, .1])
    assert scores[0] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == 0
    with pytest.raises(ValueError):
        fused_scores(["note"], "query", [float("nan")])


def test_semantic_paraphrase_beats_recent_noise_and_retains_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    current = session(tmp_path)
    remember(current, "facts", "令牌保存于保险库", source="tool_observation", observation_index=4)
    for index in range(60):
        remember(current, "facts", f"已阅读其他文件 {index}.py", source="model_note")
    before = current.model_dump_json()
    memory = asyncio.run(retrieve_memory(
        current, "登录凭据存放在哪里", index=SemanticIndex(FakeEncoder()),
    ))
    assert memory["facts"][0] == "令牌保存于保险库"
    assert memory["retrieval_status"] == "hybrid_local"
    assert memory["provenance"]["facts"][0]["observation_index"] == 4
    assert current.model_dump_json() == before


def test_controller_uses_hybrid_projection_and_audits_mode(tmp_path, monkeypatch):
    from veripatch import studio_embeddings
    from veripatch.studio_agent import StudioAgent
    from veripatch.studio_domain import StudioDecision, StudioReply
    from veripatch.studio_store import StudioStore

    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    monkeypatch.setattr(studio_embeddings, "_INDEX", SemanticIndex(FakeEncoder()))

    class Model:
        contexts = []

        async def decide(self, context):
            self.contexts.append(context)
            return StudioReply(decision=StudioDecision(action="respond", rationale="回答",
                                                       message="已读取历史笔记"))

    current = session(tmp_path)
    remember(current, "facts", "令牌保存于保险库", source="model_note")
    for number in range(45):
        remember(current, "facts", f"无关颜色配置 {number}", source="model_note")
    store = StudioStore(tmp_path / "memory.db")
    store.save(current, "created", {})
    # Start a fresh controller from persisted notes, not from a previous cache.
    restored = store.load(current.session_id)
    model = Model()
    agent = StudioAgent(model, store)
    asyncio.run(agent.handle(restored, "登录凭据存放在哪里"))
    memory = model.contexts[0]["structured_memory"]
    assert memory["retrieval_status"] == "hybrid_local"
    assert memory["facts"][0] == "令牌保存于保险库"
    recovered = agent._recovery_context(restored, model.contexts[0])
    assert recovered["structured_memory"]["retrieval_status"] == "hybrid_local"
    assert any(event["event_type"] == "memory_retrieval"
               for event in store.events(current.session_id))


def test_cache_incremental_removal_and_session_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    encoder = FakeEncoder()
    index = SemanticIndex(encoder)
    first = session(tmp_path, "first")
    remember(first, "facts", "令牌保存于保险库", source="model_note")
    query = "登录凭据存放在哪里"

    async def run():
        await retrieve_memory(first, query, index=index)
        await retrieve_memory(first, query, index=index)
        assert len(encoder.calls) == 1
        remember(first, "facts", "新笔记", source="model_note")
        await retrieve_memory(first, query, index=index)
        assert encoder.calls[-1] == ["新笔记"]
        forget_model_note(first, "facts", "令牌保存于保险库")
        await retrieve_memory(first, query, index=index)
        assert "令牌保存于保险库" not in next(iter(index._cache.values()))
        second = session(tmp_path, "second")
        memory = await retrieve_memory(second, query, index=index)
        assert memory["facts"] == []
        remember(second, "facts", "其他会话的笔记", source="model_note")
        memory = await retrieve_memory(second, query, index=index)
        assert "新笔记" not in memory["facts"]
        assert len(index._cache) == 2
        # Same ID in another workspace also cannot share a note cache.
        other = session(tmp_path / "other", "first")
        remember(other, "facts", "新笔记", source="model_note")
        await retrieve_memory(other, query, index=index)
        assert "新笔记" in encoder.calls[-1]

    asyncio.run(run())


@pytest.mark.parametrize("bad", [[[float("nan")]], [[0., 0.]], [[1.]], [[1., 0.], [1.]]])
def test_invalid_embeddings_degrade_without_losing_notes(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")

    class Broken:
        def encode(self, texts):
            return bad

    current = session(tmp_path)
    remember(current, "facts", "billing.py 退款逻辑", source="model_note")
    memory = asyncio.run(retrieve_memory(current, "billing.py", index=SemanticIndex(Broken())))
    assert memory["retrieval_status"] == "bm25_fallback:embedding_unavailable"
    assert memory["facts"][0] == "billing.py 退款逻辑"


def test_disabled_or_missing_backend_and_redaction(tmp_path, monkeypatch):
    class Unavailable:
        def encode(self, texts):
            assert all("abcdefghi" not in text for text in texts)
            raise RuntimeError("secret raw provider error abcdefghi")

    current = session(tmp_path)
    current.memory.facts = ["api_key=abcdefghi 令牌存储"]
    index = SemanticIndex(Unavailable())
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    memory = asyncio.run(retrieve_memory(current, "令牌 password=abcdefghi", index=index))
    assert "abcdefghi" not in json.dumps(memory)
    assert memory["retrieval_status"] == "bm25_fallback:embedding_unavailable"
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "bm25")
    memory = asyncio.run(retrieve_memory(current, "令牌", index=index))
    assert memory["retrieval_status"] == "bm25_configured"


def test_timeout_busy_and_cancel_remain_responsive(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    started = threading.Event()
    release = threading.Event()

    class Slow(FakeEncoder):
        def encode(self, texts):
            started.set()
            assert release.wait(timeout=5)
            return super().encode(texts)

    index = SemanticIndex(Slow())
    current = session(tmp_path)
    remember(current, "facts", "令牌保存于保险库", source="model_note")

    async def run():
        pending = asyncio.create_task(retrieve_memory(current, "查询", index=index, timeout=.05))
        while not started.is_set():
            await asyncio.sleep(.001)
        busy = await retrieve_memory(current, "查询", index=index)
        assert busy["retrieval_status"] == "bm25_fallback:embedding_busy"
        timed = await pending
        assert timed["retrieval_status"] == "bm25_fallback:embedding_timeout"
        release.set()
        while not index._gate.acquire(blocking=False):
            await asyncio.sleep(.001)
        index._gate.release()
        started.clear()
        release.clear()
        pending = asyncio.create_task(retrieve_memory(current, "另一查询", index=index))
        while not started.is_set():
            await asyncio.sleep(.001)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()

    try:
        asyncio.run(run())
    finally:
        release.set()
        index._worker.shutdown(wait=True)


def test_projection_budget_with_hybrid_metadata(tmp_path):
    current = session(tmp_path)
    for number in range(80):
        remember(current, "facts", f"{number} " + "测试笔记 " * 80, source="model_note")
    before = current.model_dump_json()
    for budget in [1000, 3000, 9000]:
        memory = project_memory(current, "测试", max_chars=budget,
                                semantic=[.8] * 80, retrieval_status="hybrid_local")
        assert len(json.dumps(memory, ensure_ascii=False)) <= budget
    assert current.model_dump_json() == before


def test_model_cache_is_bounded_and_changed_snapshot_is_not_mixed(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    started = threading.Event()
    release = threading.Event()

    class Waiting(FakeEncoder):
        def encode(self, texts):
            started.set()
            assert release.wait(5)
            return super().encode(texts)

    index = SemanticIndex(Waiting())
    current = session(tmp_path)
    remember(current, "facts", "令牌保存于保险库", source="model_note")

    async def run():
        pending = asyncio.create_task(retrieve_memory(current, "登录凭据存放在哪里", index=index))
        while not started.is_set():
            await asyncio.sleep(.001)
        forget_model_note(current, "facts", "令牌保存于保险库")
        remember(current, "facts", "更新后的笔记", source="model_note")
        release.set()
        memory = await pending
        assert memory["facts"] == ["令牌保存于保险库"]
        assert current.memory.facts == ["更新后的笔记"]
        for number in range(12):
            other = session(tmp_path, str(number))
            remember(other, "facts", str(number), source="model_note")
            await retrieve_memory(other, "查询", index=index)
        assert len(index._cache) == 8
        assert all(len(values) <= 241 for values in index._cache.values())

    try:
        asyncio.run(run())
    finally:
        release.set()
        index._worker.shutdown(wait=True)


def test_real_offline_multilingual_model(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIPATCH_MEMORY_RETRIEVAL", "hybrid")
    if not (model_directory() / "tokenizer.json").exists():
        pytest.skip("Run scripts/prepare_memory_model.py for offline model integration")
    verify_artifacts(model_directory())
    encoder = LocalEncoder()
    current = session(tmp_path)
    note = "访问令牌存储于操作系统的安全保险库。"
    query = "Where are authentication credentials kept?"
    assert bm25_scores([note], query) == [0.]
    remember(current, "facts", note, source="model_note")
    for number in range(60):
        remember(current, "facts", f"阅读了界面颜色配置 {number}", source="model_note")
    before = time.monotonic()
    memory = asyncio.run(retrieve_memory(current, query, index=SemanticIndex(encoder)))
    assert memory["retrieval_status"] == "hybrid_local"
    assert memory["facts"][0] == note
    assert time.monotonic() - before < 15

    remember(current, "relevant_files", "src/billing.py", source="tool_observation")
    memory = asyncio.run(retrieve_memory(
        current, "修改 src/billing.py", index=SemanticIndex(encoder),
    ))
    assert memory["relevant_files"][0] == "src/billing.py"
