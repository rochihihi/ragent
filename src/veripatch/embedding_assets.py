"""Pinned, data-only multilingual embedding artifacts (never execute remote code)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "2c4055b12046f11709e9df2c122e59ffbdc2f900"
MODEL_KEY = f"{MODEL_ID}@{REVISION}:int8:mean:128"
ARTIFACTS = {
    "onnx/model_quantized.onnx": (
        118308126, "66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc",
    ),
    "tokenizer.json": (
        17082913, "b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441",
    ),
}


def model_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "memory-model"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2] / "build" / "memory-model"


def verify_artifacts(directory: Path) -> None:
    for name, (size, digest) in ARTIFACTS.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("embedding_artifact_missing_or_invalid")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError("embedding_artifact_checksum_mismatch")
