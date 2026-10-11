"""Offline, bounded semantic retrieval. No provider API or cross-session search."""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from veripatch.embedding_assets import MODEL_KEY, model_directory, verify_artifacts
from veripatch.studio_domain import StudioSession
from veripatch.studio_memory import LIMITS, _clean, project_memory


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class LocalEncoder:
    """Load only checksum-verified ONNX/tokenizer data, on CPU with two threads."""

    def __init__(self, directory: Path | None = None) -> None:
        directory = directory or model_directory()
        verify_artifacts(directory)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "onnx/model_quantized.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        result: list[list[float]] = []
        for start in range(0, len(texts), 8):
            batch = self.tokenizer.encode_batch(texts[start:start + 8])
            inputs = {
                "input_ids": np.array([row.ids for row in batch], dtype=np.int64),
                "attention_mask": np.array([row.attention_mask for row in batch], dtype=np.int64),
                "token_type_ids": np.array([row.type_ids for row in batch], dtype=np.int64),
            }
            feeds = {item.name: inputs[item.name] for item in self.session.get_inputs()}
            tokens = self.session.run(None, feeds)[0]
            mask = inputs["attention_mask"][..., None].astype(np.float32)
            vectors = (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            result.extend(vectors.tolist())
        return result


def normalized_vectors(values: list[list[float]], expected: int) -> list[tuple[float, ...]]:
    if len(values) != expected:
        raise ValueError("embedding_count_mismatch")
    result: list[tuple[float, ...]] = []
    dimension = len(values[0]) if values else 0
    if not 1 <= dimension <= 4096:
        raise ValueError("embedding_dimension_invalid")
    for row in values:
        if len(row) != dimension or not all(math.isfinite(value) for value in row):
            raise ValueError("embedding_values_invalid")
        norm = math.sqrt(sum(value * value for value in row))
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("embedding_norm_invalid")
        result.append(tuple(value / norm for value in row))
    return result


class SemanticIndex:
    """One bounded worker/model; at most eight session-local ephemeral caches.

    A busy/slow worker never queues unbounded jobs or blocks the UI. Timed-out or
    cancelled inference can finish in the background, but cannot alter a session.
    Raw model errors never enter prompts/logs (they could contain paths or notes).
    """

    def __init__(self, encoder: Encoder | None = None) -> None:
        self.encoder = encoder
        self.model_key = MODEL_KEY
        self._gate = threading.BoundedSemaphore(1)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-embedding")
        self._cache: OrderedDict[tuple[str, str], dict[str, tuple[float, ...]]] = OrderedDict()
        self._retry_after = 0.0

    def _rank(self, scope: tuple[str, str], documents: tuple[str, ...], query: str) -> list[float]:
        if self.encoder is None:
            if time.monotonic() < self._retry_after:
                raise RuntimeError("embedding_load_cooldown")
            try:
                self.encoder = LocalEncoder()
            except Exception:
                self._retry_after = time.monotonic() + 60
                raise
        previous = self._cache.get(scope, {})
        # Prune removed/corrected notes and old queries. No shared document index.
        texts = list(dict.fromkeys([*documents, query]))
        missing = [text for text in texts if text not in previous]
        added = normalized_vectors(self.encoder.encode(missing), len(missing)) if missing else []
        current = {text: previous[text] for text in texts if text in previous}
        current.update(zip(missing, added, strict=True))
        dimensions = {len(row) for row in current.values()}
        if len(dimensions) != 1:
            raise ValueError("embedding_cache_dimension_mismatch")
        self._cache[scope] = current
        self._cache.move_to_end(scope)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        query_vector = current[query]
        return [sum(a * b for a, b in zip(current[text], query_vector, strict=True))
                for text in documents]

    async def scores(
        self, scope: tuple[str, str], documents: tuple[str, ...], query: str,
        *, timeout: float = 15.0,
    ) -> tuple[list[float] | None, str]:
        if not self._gate.acquire(blocking=False):
            return None, "bm25_fallback:embedding_busy"
        try:
            future = self._worker.submit(self._rank, scope, documents, query)
        except BaseException:
            self._gate.release()
            raise
        future.add_done_callback(lambda _: self._gate.release())
        try:
            values = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
            return values, "hybrid_local"
        except TimeoutError:
            return None, "bm25_fallback:embedding_timeout"
        except Exception:
            return None, "bm25_fallback:embedding_unavailable"


_INDEX = SemanticIndex()


async def retrieve_memory(
    session: StudioSession, query: str = "", *, index: SemanticIndex | None = None,
    timeout: float = 15.0, max_chars: int = 9_000,
) -> dict[str, object]:
    # Freeze notes before awaiting. A steering/edit operation cannot change the
    # association between vector positions, notes and evidence in this request.
    snapshot = session.model_copy(update={"memory": session.memory.model_copy(deep=True)})
    query = _clean(query or next(
        (item.content for item in reversed(session.messages) if item.role == "user"), "",
    ))
    documents = tuple(_clean(value) for category in LIMITS
                      for value in getattr(snapshot.memory, category))
    if os.environ.get("VERIPATCH_MEMORY_RETRIEVAL", "hybrid").lower() == "bm25":
        return project_memory(snapshot, query, max_chars=max_chars,
                              retrieval_status="bm25_configured")
    if not documents or not query:
        return project_memory(snapshot, query, max_chars=max_chars,
                              retrieval_status="bm25:no_query_or_notes")
    # A process restart rebuilds vectors from this session's persisted notes.
    scope = (str(Path(session.repo_root).resolve()), session.session_id)
    scores, status = await (index or _INDEX).scores(scope, documents, query, timeout=timeout)
    return project_memory(snapshot, query, max_chars=max_chars, semantic=scores,
                          retrieval_status=status)
