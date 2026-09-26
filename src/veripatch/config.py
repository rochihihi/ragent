"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} cannot be negative, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings with conservative defaults."""

    model: str = "gpt-5.6-terra"
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_low_balance: float = 5.0
    reasoning_effort: str = "medium"
    max_steps: int = 12
    max_model_calls: int = 12
    max_input_tokens: int = 500_000
    max_output_tokens: int = 50_000
    test_timeout_seconds: int = 120
    database_path: Path = Path("runs/veripatch.sqlite3")
    max_file_bytes: int = 512_000
    max_observations_in_prompt: int = 12
    test_runner: str = "docker"
    docker_image: str = "veripatch-sandbox:latest"
    docker_cpus: str = "1.0"
    docker_memory: str = "512m"
    docker_pids: int = 128

    @classmethod
    def from_env(cls) -> Settings:
        effort = os.getenv("VERIPATCH_REASONING_EFFORT", "medium")
        allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
        if effort not in allowed_efforts:
            raise ValueError(
                "VERIPATCH_REASONING_EFFORT must be one of "
                f"{sorted(allowed_efforts)}, got {effort!r}"
            )
        test_runner = os.getenv("VERIPATCH_TEST_RUNNER", "docker")
        if test_runner not in {"local", "docker"}:
            raise ValueError("VERIPATCH_TEST_RUNNER must be 'local' or 'docker'")
        return cls(
            model=os.getenv("VERIPATCH_MODEL", "gpt-5.6-terra"),
            openai_base_url=os.getenv("VERIPATCH_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            deepseek_model=os.getenv("VERIPATCH_DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_base_url=os.getenv("VERIPATCH_DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_low_balance=_nonnegative_float("VERIPATCH_DEEPSEEK_LOW_BALANCE", 5.0),
            reasoning_effort=effort,
            max_steps=_positive_int("VERIPATCH_MAX_STEPS", 12),
            max_model_calls=_positive_int("VERIPATCH_MAX_MODEL_CALLS", 12),
            max_input_tokens=_positive_int("VERIPATCH_MAX_INPUT_TOKENS", 500_000),
            max_output_tokens=_positive_int("VERIPATCH_MAX_OUTPUT_TOKENS", 50_000),
            test_timeout_seconds=_positive_int("VERIPATCH_TEST_TIMEOUT_SECONDS", 120),
            database_path=Path(os.getenv("VERIPATCH_DATABASE_PATH", "runs/veripatch.sqlite3")),
            max_file_bytes=_positive_int("VERIPATCH_MAX_FILE_BYTES", 512_000),
            max_observations_in_prompt=_positive_int("VERIPATCH_MAX_OBSERVATIONS_IN_PROMPT", 12),
            test_runner=test_runner,
            docker_image=os.getenv("VERIPATCH_DOCKER_IMAGE", "veripatch-sandbox:latest"),
            docker_cpus=os.getenv("VERIPATCH_DOCKER_CPUS", "1.0"),
            docker_memory=os.getenv("VERIPATCH_DOCKER_MEMORY", "512m"),
            docker_pids=_positive_int("VERIPATCH_DOCKER_PIDS", 128),
        )
